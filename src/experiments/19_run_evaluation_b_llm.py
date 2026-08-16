#!/usr/bin/env python3
"""Run frozen M3/M4 AMP extraction for the retained Evaluation-B cases.

This runner is intentionally separate from the frozen A1/A2 namespaces.  It
reuses the frozen request builder and technical retry policy, sends no human or
silver labels, and persists one atomic success record per case before
materializing a canonical JSONL file.  A prior response is reused only when its
complete frozen request hash is identical to the newly rebuilt request.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE4_RUNNER_PATH = REPO_ROOT / "src/experiments/10_run_llm_amp.py"
DEFAULT_REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
DEFAULT_SAMPLE = REPO_ROOT / "data/annotations/reliability_sample_100.csv"
DEFAULT_RAW_ANNOTATIONS = REPO_ROOT / "data/annotations/reviewer_annotation_template.csv"
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT / "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
)
DEFAULT_QC_SUMMARY = (
    REPO_ROOT / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
)
DEFAULT_MEMBERSHIP_MANIFEST = (
    REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_membership_manifest.json"
)
DEFAULT_LEAKAGE_AUDIT = (
    REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv"
)
DEFAULT_MODEL_MARKER = REPO_ROOT / "outputs/logs/llm/model_access.json"
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions/evaluation_b"
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/evaluation_b/llm"
EXPECTED_RAW_SOURCE_SHA256 = (
    "7ec0a40ab6a9d64588cf4b6c8b46d2572683cf7e340d786117604bc6f20081af"
)
EXPECTED_SOURCE_ROWS = 100
ALLOWED_REVIEW_STATUSES = frozenset({"SUBSTANTIVE", "ABSTAIN"})


class EvaluationBLLMError(RuntimeError):
    """Raised before a request when an Evaluation-B invariant is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBLLMError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationBLLMError(f"JSON artifact is not an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise EvaluationBLLMError(f"Cannot read CSV artifact {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    line_number = 0
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("row is not an object")
            rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationBLLMError(
            f"Cannot read JSONL artifact {path} at line {line_number}: {exc}"
        ) from exc
    return rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any], *, secret: str | None = None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if secret and secret in text:
        raise EvaluationBLLMError("Credential material reached a serialized artifact")
    _atomic_text(path, text)


def atomic_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, secret: str | None = None
) -> None:
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    if secret and secret in text:
        raise EvaluationBLLMError("Credential material reached a serialized artifact")
    _atomic_text(path, text)


def load_phase4_runner() -> ModuleType:
    """Load the numeric-name frozen runner without executing its CLI."""

    name = "_sherloc_phase4_llm_runner_for_evaluation_b"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, PHASE4_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise EvaluationBLLMError(f"Cannot load frozen runner: {PHASE4_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _first(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def retained_membership_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        {
            "reliability_case_id": str(case["reliability_case_id"]),
            "search_rank": int(case["search_rank"]),
            "canonical_url": str(case["canonical_url"]),
            "input_sha256": sha256_text(str(case["fact_summary"])),
            "review_status": str(case["review_status"]),
        }
        for case in sorted(cases, key=lambda item: int(item["search_rank"]))
    ]
    return sha256_text(canonical_json(payload))


def load_retained_cases(
    reference_path: Path = DEFAULT_REFERENCE,
    sample_path: Path = DEFAULT_SAMPLE,
) -> list[dict[str, Any]]:
    references = load_csv(reference_path)
    samples = load_csv(sample_path)
    sample_by_id: dict[str, dict[str, str]] = {}
    for row in samples:
        case_id = str(row.get("reliability_case_id") or "").strip()
        if not case_id or case_id in sample_by_id:
            raise EvaluationBLLMError("Reliability sample IDs are blank or duplicated")
        sample_by_id[case_id] = row
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for row in references:
        reliability_id = str(row.get("reliability_case_id") or "").strip()
        if not reliability_id or reliability_id in seen_ids:
            raise EvaluationBLLMError("Human-reference IDs are blank or duplicated")
        if reliability_id not in sample_by_id:
            raise EvaluationBLLMError(
                f"Human-reference ID is absent from reliability sample: {reliability_id}"
            )
        sample = sample_by_id[reliability_id]
        try:
            rank = int(_first(row, ("search_rank",)) or sample["search_rank"])
        except (KeyError, ValueError) as exc:
            raise EvaluationBLLMError(f"Invalid search rank for {reliability_id}") from exc
        if rank in seen_ranks or rank != int(sample["search_rank"]):
            raise EvaluationBLLMError(f"Search-rank mismatch/duplicate for {reliability_id}")
        status = str(row.get("review_status") or "").strip().upper()
        if status not in ALLOWED_REVIEW_STATUSES:
            raise EvaluationBLLMError(
                f"Human reference contains non-retained status for {reliability_id}: {status}"
            )
        fact_summary = _first(
            row, ("english_fact_summary_raw", "fact_summary", "Fact Summary")
        )
        sample_text = str(sample.get("english_fact_summary_raw") or "")
        if not fact_summary or fact_summary != sample_text:
            raise EvaluationBLLMError(
                f"Fact Summary drift between reference/sample for {reliability_id}"
            )
        canonical_url = _first(row, ("canonical_url",)) or str(
            sample.get("canonical_url") or ""
        )
        jurisdiction = _first(row, ("jurisdiction", "jurisdiction_raw")) or str(
            sample.get("jurisdiction_raw") or ""
        )
        if canonical_url != str(sample.get("canonical_url") or "") or not jurisdiction:
            raise EvaluationBLLMError(f"Identity drift for {reliability_id}")
        unodc_number = str(sample.get("unodc_case_number") or "").strip()
        cases.append(
            {
                "reliability_case_id": reliability_id,
                "case_id": unodc_number or f"sherloc-rank-{rank}",
                "search_rank": rank,
                "case_title": str(sample.get("case_title") or ""),
                "canonical_url": canonical_url,
                "jurisdiction": jurisdiction,
                "fact_summary": fact_summary,
                "review_status": status,
                "role": "EVALUATION_B_RETAINED",
            }
        )
        seen_ids.add(reliability_id)
        seen_ranks.add(rank)
    if not cases:
        raise EvaluationBLLMError("The retained human reference is empty")
    return sorted(cases, key=lambda item: int(item["search_rank"]))


def validate_human_gates(
    cases: Sequence[Mapping[str, Any]],
    *,
    reference_path: Path = DEFAULT_REFERENCE,
    raw_annotations_path: Path = DEFAULT_RAW_ANNOTATIONS,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    qc_summary_path: Path = DEFAULT_QC_SUMMARY,
    membership_manifest_path: Path = DEFAULT_MEMBERSHIP_MANIFEST,
    leakage_audit_path: Path = DEFAULT_LEAKAGE_AUDIT,
) -> dict[str, Any]:
    """Fail closed before spend on source, QC, membership, and leakage gates."""

    observed_source_sha = sha256_file(raw_annotations_path)
    if observed_source_sha != EXPECTED_RAW_SOURCE_SHA256:
        raise EvaluationBLLMError("Immutable raw human annotation SHA-256 changed")
    source = load_json(source_manifest_path)
    source_sha = _first(source, ("sha256", "source_sha256", "file_sha256"))
    source_rows = source.get("row_count", source.get("source_row_count"))
    if source_sha != observed_source_sha or int(source_rows or -1) != EXPECTED_SOURCE_ROWS:
        raise EvaluationBLLMError("Human annotation source manifest is stale")
    qc = load_json(qc_summary_path)
    blocking = int(
        qc.get(
            "blocking_error_count",
            qc.get("material_scoring_contradiction_count", qc.get("fatal_issue_count", -1)),
        )
    )
    if blocking != 0:
        raise EvaluationBLLMError(f"Human annotation QC has {blocking} blocking issues")
    expected_membership = retained_membership_sha256(cases)
    membership = load_json(membership_manifest_path)
    observed_membership = _first(
        membership,
        (
            "retained_membership_sha256",
            "membership_sha256",
            "eval_b_retained_membership_sha256",
        ),
    )
    retained_n = membership.get("retained_n", membership.get("retained_case_count"))
    overlap_audit = membership.get("a1_active_m4_demo_overlap_audit")
    expected_reference_sha = sha256_file(reference_path)
    if (
        membership.get("status") != "FROZEN_FOR_EVALUATION_B_PRE_MODEL_INFERENCE"
        or membership.get("human_reference_sha256") != expected_reference_sha
        or observed_membership != expected_membership
        or int(retained_n or -1) != len(cases)
        or not isinstance(overlap_audit, Mapping)
        or int(overlap_audit.get("overlap_n") or 0) != 0
        or overlap_audit.get("status") != "PASS_NO_OVERLAP"
    ):
        raise EvaluationBLLMError("Evaluation-B retained membership manifest is stale")
    audit = load_csv(leakage_audit_path)
    audit_by_id = {
        str(row.get("reliability_case_id") or "").strip(): row for row in audit
    }
    if len(audit_by_id) != len(audit) or set(audit_by_id) != {
        str(case["reliability_case_id"]) for case in cases
    }:
        raise EvaluationBLLMError("Leakage audit membership differs from retained cases")
    true_values = {"1", "TRUE", "YES", "Y"}
    leaked = [
        case_id
        for case_id, row in audit_by_id.items()
        if str(row.get("removed_from_eval_b_supervised_training") or "")
        .strip()
        .upper()
        not in true_values
    ]
    if leaked:
        raise EvaluationBLLMError(
            f"Retained cases are not excluded from supervised training: {leaked}"
        )
    return {
        "source_sha256": observed_source_sha,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "qc_summary_sha256": sha256_file(qc_summary_path),
        "membership_manifest_sha256": sha256_file(membership_manifest_path),
        "human_reference_sha256": expected_reference_sha,
        "retained_membership_sha256": expected_membership,
        "leakage_audit_sha256": sha256_file(leakage_audit_path),
        "retained_n": len(cases),
        "substantive_n": sum(case["review_status"] == "SUBSTANTIVE" for case in cases),
        "abstain_n": sum(case["review_status"] == "ABSTAIN" for case in cases),
    }


def load_model_marker(path: Path, phase4: ModuleType) -> dict[str, Any]:
    marker = load_json(path)
    expected = {
        "status": "MODEL_ACCESS_CONFIRMED",
        "config_sha256": phase4.EXPECTED_CONFIG_SHA256,
        "schema_sha256": phase4.builder.SCHEMA_SHA256
        if hasattr(phase4.builder, "SCHEMA_SHA256")
        else marker.get("schema_sha256"),
        "requested_model_id": phase4.MODEL_ALIAS,
    }
    mismatches = {
        key: {"expected": value, "observed": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    effective = str(marker.get("effective_model_id") or "")
    if not effective or not (
        effective == phase4.MODEL_ALIAS or phase4.DATED_MODEL_PATTERN.fullmatch(effective)
    ):
        mismatches["effective_model_id"] = {
            "expected": "frozen alias or dated snapshot",
            "observed": effective,
        }
    if mismatches:
        raise EvaluationBLLMError(
            "Model-access marker is stale: " + canonical_json(mismatches)
        )
    marker["marker_sha256"] = sha256_file(path)
    return marker


def make_spec(method: str, prediction_root: Path, log_root: Path, phase4: ModuleType) -> Any:
    normalized = method.upper()
    if normalized not in {"M3", "M4"}:
        raise EvaluationBLLMError("method must be M3 or M4")
    method_name = normalized.lower()
    return phase4.RunSpec(
        method=normalized,
        evaluation="EVALUATION_B",
        fold=None,
        dry_run=False,
        bank_id="A1" if normalized == "M4" else None,
        output_path=prediction_root / method_name / "eval_b_predictions.jsonl",
        state_dir=log_root / "state" / method_name,
        diagnostics_path=log_root / f"{method_name}_diagnostics.json",
        failure_manifest_path=log_root / f"{method_name}_failures.jsonl",
    )


def prepare_requests(
    method: str,
    cases: Sequence[dict[str, Any]],
    *,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
    model_marker_path: Path = DEFAULT_MODEL_MARKER,
) -> dict[str, Any]:
    phase4 = load_phase4_runner()
    phase4.validate_canonical_artifact_hashes()
    contract, config, bank = phase4.validate_frozen_contract()
    benchmark = phase4.load_benchmark_index()
    spec = make_spec(method, prediction_root, log_root, phase4)
    marker = load_model_marker(model_marker_path, phase4)
    demos = None
    demo_metadata = None
    if spec.method == "M4":
        demos, demo_metadata = phase4.load_demo_bank_for_setting(
            "A1", bank, config, benchmark, actual_test_jurisdictions=[]
        )
    demo_ranks = {int(item["search_rank"]) for item in (demos or [])}
    overlap = sorted(demo_ranks & {int(case["search_rank"]) for case in cases})
    if overlap:
        raise EvaluationBLLMError(
            f"Retained Evaluation-B cases overlap frozen A1 M4 demonstrations: {overlap}"
        )
    requests: dict[int, dict[str, Any]] = {}
    for case in cases:
        request = phase4.build_request_for_case(
            spec,
            case,
            demos=demos,
            demo_metadata=demo_metadata,
            heldout_jurisdictions=[],
            effective_model_id=str(marker["effective_model_id"]),
            contract=contract,
            config=config,
            config_path=phase4.DEFAULT_CONFIG,
            m3_prompt_path=phase4.DEFAULT_M3_PROMPT,
            m4_prompt_path=phase4.DEFAULT_M4_PROMPT,
        )
        payload = request["payload"]
        if payload.get("store") is not False:
            raise EvaluationBLLMError("Frozen request must retain store=false")
        if payload.get("max_output_tokens") != phase4.INITIAL_MAX_OUTPUT_TOKENS:
            raise EvaluationBLLMError("Frozen request must start at max_output_tokens=512")
        messages = payload.get("input")
        expected_target = {
            "role": "user",
            "content": "Extract this case from the supplied evidence only:\n"
            + canonical_json({"fact_summary": case["fact_summary"]}),
        }
        if (
            not isinstance(messages, list)
            or not messages
            or messages[-1] != expected_target
        ):
            raise EvaluationBLLMError(
                "Frozen target payload must contain only the exact Fact Summary"
            )
        rank = int(case["search_rank"])
        requests[rank] = request
    return {
        "phase4": phase4,
        "spec": spec,
        "contract": contract,
        "config": config,
        "demo_bank": bank,
        "benchmark": benchmark,
        "model_marker": marker,
        "demos": demos,
        "demo_metadata": demo_metadata,
        "demo_overlap_ranks": overlap,
        "cases": list(cases),
        "requests": requests,
        "retained_membership_sha256": retained_membership_sha256(cases),
    }


def _prior_prediction_paths(method: str) -> list[Path]:
    root = REPO_ROOT / "outputs/predictions" / method.lower()
    if method.upper() == "M4":
        return [root / "a1_test_predictions.jsonl"]
    return sorted(root.glob("a1_test_predictions.jsonl")) + sorted(
        root.glob("a2_fold_*_test_predictions.jsonl")
    )


def reusable_prediction_index(
    method: str,
    requests: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    reusable: dict[int, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for path in _prior_prediction_paths(method):
        if not path.is_file():
            continue
        file_sha = sha256_file(path)
        for line_number, record in enumerate(load_jsonl(path), start=1):
            try:
                rank = int(record.get("search_rank"))
            except (TypeError, ValueError):
                continue
            request = requests.get(rank)
            if request is None:
                continue
            reasons: list[str] = []
            if record.get("status") != "SUCCESS_VALIDATED":
                reasons.append("status")
            if str(record.get("method") or "").upper() != method.upper():
                reasons.append("method")
            if record.get("request_sha256") != request.get("request_sha256"):
                reasons.append("request_sha256")
            if not isinstance(record.get("validated_prediction"), Mapping):
                reasons.append("validated_prediction")
            if reasons:
                rejected.append(
                    {
                        "search_rank": rank,
                        "source_path": display_path(path),
                        "reasons": reasons,
                    }
                )
                continue
            candidate = {
                "record": record,
                "source_path": display_path(path),
                "source_file_sha256": file_sha,
                "source_line_number": line_number,
                "source_record_sha256": sha256_text(canonical_json(record)),
            }
            # Deterministic first path/line wins when identical M3 requests recur.
            reusable.setdefault(rank, candidate)
    return reusable, rejected


def _success_path(spec: Any, rank: int) -> Path:
    return spec.state_dir / "success" / f"{rank:06d}.json"


def _failure_path(spec: Any, rank: int) -> Path:
    return spec.state_dir / "failures" / f"{rank:06d}.json"


def make_reused_record(
    method: str,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    reusable: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    source = reusable["record"]
    prediction = prepared["phase4"].builder.validate_structured_output(
        source["validated_prediction"]
    )
    return {
        "schema_version": "sherloc-evaluation-b-llm-prediction-v1",
        "runner_version": VERSION,
        "status": "SUCCESS_VALIDATED",
        "method": method.upper(),
        "evaluation": "B",
        "subset": "RETAINED",
        "reliability_case_id": case["reliability_case_id"],
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "case_title": case["case_title"],
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": sha256_text(str(case["fact_summary"])),
        "review_status": case["review_status"],
        "predicted_labels": prediction["acts"] + prediction["means"] + prediction["purposes"],
        "normalized_prediction": prediction,
        "validated_prediction": prediction,
        "raw_structured_response": source.get("raw_structured_response"),
        "raw_structured_response_text": source.get("raw_structured_response_text"),
        "request_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "prompt_sha256": prepared["config"]["methods"][method.upper()]["prompt_sha256"],
        "schema_sha256": prepared["config"]["structured_output"]["schema_sha256"],
        "demo_bank_id": (prepared.get("demo_metadata") or {}).get("demo_bank_id"),
        "demo_bank_membership_sha256": (prepared.get("demo_metadata") or {}).get(
            "demo_bank_membership_sha256"
        ),
        "requested_model_id": prepared["phase4"].MODEL_ALIAS,
        "effective_requested_model_id": prepared["model_marker"]["effective_model_id"],
        "api_request_issued_for_evaluation_b": False,
        "reuse_status": "REUSED_IDENTICAL_FROZEN_REQUEST",
        "reuse_source_path": reusable["source_path"],
        "reuse_source_file_sha256": reusable["source_file_sha256"],
        "reuse_source_line_number": reusable["source_line_number"],
        "reuse_source_record_sha256": reusable["source_record_sha256"],
        "source_response_id": source.get("response_id"),
        "source_execution_timestamp": source.get("execution_timestamp"),
        "source_token_usage": source.get("token_usage"),
        "token_usage": None,
        "latency_seconds": 0.0,
        "retry_count": 0,
        "retry_events": [],
        "technical_execution": None,
        "execution_timestamp": utc_now(),
        "retained_membership_sha256": prepared["retained_membership_sha256"],
        "human_or_silver_labels_sent_to_model": False,
        "store": False,
    }


def make_api_success_record(
    method: str,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    sdk_version: str,
) -> dict[str, Any]:
    phase4 = prepared["phase4"]
    parsed = result["parsed"]
    prediction = parsed["validated_prediction"]
    response = result["response"]
    technical = phase4._validated_technical_execution_provenance(result, request)
    return {
        "schema_version": "sherloc-evaluation-b-llm-prediction-v1",
        "runner_version": VERSION,
        "status": "SUCCESS_VALIDATED",
        "method": method.upper(),
        "evaluation": "B",
        "subset": "RETAINED",
        "reliability_case_id": case["reliability_case_id"],
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "case_title": case["case_title"],
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": sha256_text(str(case["fact_summary"])),
        "review_status": case["review_status"],
        "predicted_labels": prediction["acts"] + prediction["means"] + prediction["purposes"],
        "normalized_prediction": prediction,
        "validated_prediction": prediction,
        "raw_structured_response": parsed["raw_structured_response"],
        "raw_structured_response_text": parsed["raw_structured_response_text"],
        "request_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "prompt_sha256": prepared["config"]["methods"][method.upper()]["prompt_sha256"],
        "schema_sha256": prepared["config"]["structured_output"]["schema_sha256"],
        "demo_bank_id": (prepared.get("demo_metadata") or {}).get("demo_bank_id"),
        "demo_bank_membership_sha256": (prepared.get("demo_metadata") or {}).get(
            "demo_bank_membership_sha256"
        ),
        "requested_model_id": phase4.MODEL_ALIAS,
        "effective_requested_model_id": prepared["model_marker"]["effective_model_id"],
        "returned_model_id": str(phase4._object_attr(response, "model") or "") or None,
        "response_id": str(phase4._object_attr(response, "id") or "") or None,
        "api_request_issued_for_evaluation_b": True,
        "reuse_status": "NEW_EVALUATION_B_REQUEST",
        "token_usage": phase4._usage(response),
        "latency_seconds": float(result["latency_seconds"]),
        "retry_count": int(result["retry_count"]),
        "retry_events": result["retry_events"],
        "technical_execution": technical,
        "sdk_version": sdk_version,
        "execution_timestamp": utc_now(),
        "retained_membership_sha256": prepared["retained_membership_sha256"],
        "human_or_silver_labels_sent_to_model": False,
        "store": False,
    }


def _load_valid_success(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    method: str,
    prepared: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = load_json(path)
    expected = {
        "schema_version": "sherloc-evaluation-b-llm-prediction-v1",
        "status": "SUCCESS_VALIDATED",
        "method": method.upper(),
        "evaluation": "B",
        "subset": "RETAINED",
        "reliability_case_id": case["reliability_case_id"],
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "case_title": case["case_title"],
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": sha256_text(str(case["fact_summary"])),
        "review_status": case["review_status"],
        "request_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "prompt_sha256": prepared["config"]["methods"][method.upper()][
            "prompt_sha256"
        ],
        "schema_sha256": prepared["config"]["structured_output"]["schema_sha256"],
        "demo_bank_id": (prepared.get("demo_metadata") or {}).get("demo_bank_id"),
        "demo_bank_membership_sha256": (prepared.get("demo_metadata") or {}).get(
            "demo_bank_membership_sha256"
        ),
        "requested_model_id": prepared["phase4"].MODEL_ALIAS,
        "effective_requested_model_id": prepared["model_marker"][
            "effective_model_id"
        ],
        "retained_membership_sha256": prepared["retained_membership_sha256"],
        "human_or_silver_labels_sent_to_model": False,
        "store": False,
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise EvaluationBLLMError(
            f"Existing Evaluation-B success is stale at {path}: {canonical_json(mismatches)}"
        )
    try:
        prediction = prepared["phase4"].builder.validate_structured_output(
            record.get("validated_prediction")
        )
    except Exception as exc:
        raise EvaluationBLLMError(
            f"Existing Evaluation-B success has an invalid prediction at {path}: {exc}"
        ) from exc
    expected_labels = (
        prediction["acts"] + prediction["means"] + prediction["purposes"]
    )
    if (
        record.get("validated_prediction") != prediction
        or record.get("normalized_prediction") != prediction
        or record.get("predicted_labels") != expected_labels
    ):
        raise EvaluationBLLMError(
            f"Existing Evaluation-B success has inconsistent predictions at {path}"
        )
    issued = record.get("api_request_issued_for_evaluation_b")
    reuse_status = record.get("reuse_status")
    if not (
        (issued is True and reuse_status == "NEW_EVALUATION_B_REQUEST")
        or (issued is False and reuse_status == "REUSED_IDENTICAL_FROZEN_REQUEST")
    ):
        raise EvaluationBLLMError(
            f"Existing Evaluation-B success has invalid request provenance at {path}"
        )
    return record


def failure_resume_policy(
    path: Path,
    phase4: ModuleType,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    policy = {
        "start_with_fallback": False,
        "prior_fallback_attempts": 0,
        "prior_primary_incomplete_provenance": None,
    }
    if not path.is_file():
        return policy
    failure = load_json(path)
    expected_wrapper = {
        "schema_version": "sherloc-evaluation-b-llm-failure-history-v1",
        "method": method.upper(),
        "reliability_case_id": case["reliability_case_id"],
        "search_rank": int(case["search_rank"]),
        "request_sha256": request["request_sha256"],
    }
    wrapper_mismatches = [
        field
        for field, expected in expected_wrapper.items()
        if failure.get(field) != expected
    ]
    if wrapper_mismatches:
        raise EvaluationBLLMError(
            f"Failure history identity drift at {path}: {wrapper_mismatches}"
        )
    history = failure.get("attempt_history")
    if not isinstance(history, list) or not history:
        raise EvaluationBLLMError(f"Malformed failure history: {path}")
    proof: Mapping[str, Any] | None = None
    prior = 0
    for attempt in history:
        if not isinstance(attempt, Mapping):
            raise EvaluationBLLMError(f"Malformed failure history entry: {path}")
        expected_attempt = {
            "status": "FAILED_NO_PREDICTION",
            "method": method.upper(),
            "evaluation": "B",
            "reliability_case_id": case["reliability_case_id"],
            "search_rank": int(case["search_rank"]),
            "canonical_url": case["canonical_url"],
            "input_sha256": sha256_text(str(case["fact_summary"])),
            "request_sha256": request["request_sha256"],
        }
        attempt_mismatches = [
            field
            for field, expected in expected_attempt.items()
            if attempt.get(field) != expected
        ]
        if attempt_mismatches:
            raise EvaluationBLLMError(
                f"Failure attempt identity drift at {path}: {attempt_mismatches}"
            )
        technical = attempt.get("technical_execution")
        if not isinstance(technical, Mapping):
            raise EvaluationBLLMError(f"Malformed technical failure provenance: {path}")
        count = technical.get("cumulative_output_token_fallback_attempts", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EvaluationBLLMError(f"Invalid fallback count in {path}")
        prior = max(prior, count)
        candidate = technical.get("initial_incomplete_response_provenance")
        if candidate is not None:
            if not (
                isinstance(candidate, Mapping)
                and candidate.get("response_status") == "incomplete"
                and isinstance(candidate.get("incomplete_details"), Mapping)
                and candidate["incomplete_details"].get("reason")
                == "max_output_tokens"
                and candidate.get("max_output_tokens")
                == phase4.INITIAL_MAX_OUTPUT_TOKENS
                and candidate.get("actual_request_sha256")
                == request["request_sha256"]
            ):
                raise EvaluationBLLMError(
                    f"Failure history lacks an exact current-request 512 trigger: {path}"
                )
            proof = candidate
    if proof is not None:
        if prior >= phase4.MAX_FALLBACK_ATTEMPTS_PER_CASE:
            raise EvaluationBLLMError(
                f"Fallback attempt ceiling is exhausted for {path.stem}"
            )
        policy.update(
            {
                "start_with_fallback": True,
                "prior_fallback_attempts": prior,
                "prior_primary_incomplete_provenance": proof,
            }
        )
    elif prior:
        raise EvaluationBLLMError(
            f"Failure history has fallback calls without a valid 512 trigger: {path}"
        )
    return policy


def _record_failure(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    method: str,
    sdk_version: str,
    secret: str,
) -> None:
    phase4 = prepared["phase4"]
    record = {
        "recorded_at": utc_now(),
        "status": "FAILED_NO_PREDICTION",
        "method": method.upper(),
        "evaluation": "B",
        "reliability_case_id": case["reliability_case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "input_sha256": sha256_text(str(case["fact_summary"])),
        "request_sha256": request["request_sha256"],
        "error": result.get("error"),
        "fatal_access_error": bool(result.get("fatal_access_error")),
        "retry_events": result.get("retry_events", []),
        "technical_execution": phase4._validated_technical_execution_provenance(
            result, request
        ),
        "sdk_version": sdk_version,
    }
    history: list[Any] = []
    if path.is_file():
        existing = load_json(path)
        if existing.get("request_sha256") != request["request_sha256"]:
            raise EvaluationBLLMError(f"Failure history request drift: {path}")
        history = list(existing.get("attempt_history") or [])
    wrapper = {
        "schema_version": "sherloc-evaluation-b-llm-failure-history-v1",
        "method": method.upper(),
        "reliability_case_id": case["reliability_case_id"],
        "search_rank": int(case["search_rank"]),
        "request_sha256": request["request_sha256"],
        "attempt_history": history + [record],
    }
    atomic_json(path, wrapper, secret=secret)


def materialize(
    prepared: Mapping[str, Any],
    *,
    secret: str | None,
    started_at: str,
) -> dict[str, Any]:
    spec = prepared["spec"]
    method = spec.method
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[int] = []
    for case in prepared["cases"]:
        rank = int(case["search_rank"])
        request = prepared["requests"][rank]
        success = _load_valid_success(
            _success_path(spec, rank),
            case=case,
            request=request,
            method=method,
            prepared=prepared,
        )
        if success is not None:
            successes.append(success)
            continue
        failure_path = _failure_path(spec, rank)
        if failure_path.is_file():
            failures.append(load_json(failure_path))
        else:
            missing.append(rank)
    successes.sort(key=lambda row: int(row["search_rank"]))
    failures.sort(key=lambda row: int(row["search_rank"]))
    atomic_jsonl(spec.output_path, successes, secret=secret)
    atomic_jsonl(spec.failure_manifest_path, failures, secret=secret)
    expected = len(prepared["cases"])
    diagnostics = {
        "schema_version": "sherloc-evaluation-b-llm-diagnostics-v1",
        "runner_version": VERSION,
        "status": "COMPLETE" if len(successes) == expected else "INCOMPLETE",
        "method": method,
        "evaluation": "B",
        "started_at": started_at,
        "completed_at": utc_now(),
        "expected_cases": expected,
        "successful_predictions": len(successes),
        "reused_identical_requests": sum(
            row.get("reuse_status") == "REUSED_IDENTICAL_FROZEN_REQUEST"
            for row in successes
        ),
        "new_api_request_successes": sum(
            row.get("api_request_issued_for_evaluation_b") is True for row in successes
        ),
        "unresolved_failures": len(failures),
        "missing_unattempted": len(missing),
        "failure_ranks": [int(row["search_rank"]) for row in failures],
        "missing_ranks": missing,
        "retained_membership_sha256": prepared["retained_membership_sha256"],
        "prediction_file": display_path(spec.output_path),
        "prediction_file_sha256": sha256_file(spec.output_path),
        "prompt_sha256": prepared["config"]["methods"][method]["prompt_sha256"],
        "schema_sha256": prepared["config"]["structured_output"]["schema_sha256"],
        "model": prepared["model_marker"]["effective_model_id"],
        "store": False,
        "demo_bank_id": (prepared.get("demo_metadata") or {}).get("demo_bank_id"),
        "demo_overlap_ranks": prepared["demo_overlap_ranks"],
        "human_or_silver_labels_sent_to_model": False,
        "human_gate": prepared.get("human_gate"),
    }
    atomic_json(spec.diagnostics_path, diagnostics, secret=secret)
    return diagnostics


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            raise EvaluationBLLMError(f"Another Evaluation-B run holds {self.path}") from exc
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def validate_m3_gate(
    prediction_root: Path,
    log_root: Path,
    cases: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_sha256: str | None = None,
    expected_schema_sha256: str | None = None,
    expected_model: str | None = None,
) -> None:
    diagnostics = load_json(log_root / "m3_diagnostics.json")
    prediction_path = prediction_root / "m3/eval_b_predictions.jsonl"
    predictions = load_jsonl(prediction_path)
    expected_membership = retained_membership_sha256(cases)
    expected_ids = {str(case["reliability_case_id"]) for case in cases}
    observed_ids = {str(row.get("reliability_case_id") or "") for row in predictions}
    diagnostics_invalid = (
        diagnostics.get("status") != "COMPLETE"
        or diagnostics.get("method") != "M3"
        or diagnostics.get("evaluation") != "B"
        or int(diagnostics.get("expected_cases") or -1) != len(cases)
        or int(diagnostics.get("successful_predictions") or -1) != len(cases)
        or int(diagnostics.get("unresolved_failures") or 0) != 0
        or int(diagnostics.get("missing_unattempted") or 0) != 0
        or diagnostics.get("retained_membership_sha256") != expected_membership
        or diagnostics.get("prediction_file_sha256") != sha256_file(prediction_path)
        or diagnostics.get("store") is not False
        or diagnostics.get("human_or_silver_labels_sent_to_model") is not False
        or (
            expected_prompt_sha256 is not None
            and diagnostics.get("prompt_sha256") != expected_prompt_sha256
        )
        or (
            expected_schema_sha256 is not None
            and diagnostics.get("schema_sha256") != expected_schema_sha256
        )
        or (
            expected_model is not None and diagnostics.get("model") != expected_model
        )
    )
    if (
        diagnostics_invalid
        or len(predictions) != len(cases)
        or observed_ids != expected_ids
    ):
        raise EvaluationBLLMError("M4 is blocked until Evaluation-B M3 is complete")
    case_by_id = {str(case["reliability_case_id"]): case for case in cases}
    phase4 = load_phase4_runner()
    for row in predictions:
        case = case_by_id[str(row["reliability_case_id"])]
        expected = {
            "status": "SUCCESS_VALIDATED",
            "method": "M3",
            "evaluation": "B",
            "subset": "RETAINED",
            "search_rank": int(case["search_rank"]),
            "canonical_url": case["canonical_url"],
            "input_sha256": sha256_text(str(case["fact_summary"])),
            "review_status": case["review_status"],
            "retained_membership_sha256": expected_membership,
            "store": False,
            "human_or_silver_labels_sent_to_model": False,
        }
        if expected_prompt_sha256 is not None:
            expected["prompt_sha256"] = expected_prompt_sha256
        if expected_schema_sha256 is not None:
            expected["schema_sha256"] = expected_schema_sha256
        if expected_model is not None:
            expected["effective_requested_model_id"] = expected_model
        if any(row.get(field) != value for field, value in expected.items()):
            raise EvaluationBLLMError(
                "M4 is blocked by stale Evaluation-B M3 prediction provenance"
            )
        try:
            prediction = phase4.builder.validate_structured_output(
                row.get("validated_prediction")
            )
        except Exception as exc:
            raise EvaluationBLLMError(
                "M4 is blocked by an invalid Evaluation-B M3 prediction"
            ) from exc
        if (
            row.get("validated_prediction") != prediction
            or row.get("normalized_prediction") != prediction
            or row.get("predicted_labels")
            != prediction["acts"] + prediction["means"] + prediction["purposes"]
        ):
            raise EvaluationBLLMError(
                "M4 is blocked by inconsistent Evaluation-B M3 prediction fields"
            )


def execute(
    prepared: Mapping[str, Any],
    *,
    workers: int,
    max_attempts: int,
    base_backoff_seconds: float,
) -> dict[str, Any]:
    phase4 = prepared["phase4"]
    spec = prepared["spec"]
    method = spec.method
    cases = list(prepared["cases"])
    if workers < 1 or workers > phase4.MAX_WORKERS:
        raise EvaluationBLLMError(f"workers must be between 1 and {phase4.MAX_WORKERS}")
    if method == "M4":
        validate_m3_gate(
            spec.output_path.parents[1],
            spec.diagnostics_path.parent,
            cases,
            expected_prompt_sha256=prepared["config"]["methods"]["M3"][
                "prompt_sha256"
            ],
            expected_schema_sha256=prepared["config"]["structured_output"][
                "schema_sha256"
            ],
            expected_model=prepared["model_marker"]["effective_model_id"],
        )
    started_at = utc_now()
    spec.state_dir.joinpath("success").mkdir(parents=True, exist_ok=True)
    spec.state_dir.joinpath("failures").mkdir(parents=True, exist_ok=True)
    reusable, _rejected = reusable_prediction_index(method, prepared["requests"])
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    with RunLock(spec.state_dir / ".run.guard"):
        for case in cases:
            rank = int(case["search_rank"])
            request = prepared["requests"][rank]
            success_path = _success_path(spec, rank)
            if _load_valid_success(
                success_path,
                case=case,
                request=request,
                method=method,
                prepared=prepared,
            ) is not None:
                continue
            if rank in reusable:
                atomic_json(
                    success_path,
                    make_reused_record(method, case, request, reusable[rank], prepared),
                )
                continue
            resume = failure_resume_policy(
                _failure_path(spec, rank),
                phase4,
                case=case,
                request=request,
                method=method,
            )
            pending.append((case, request, resume))

        # Rebuild every pending request after one final frozen-artifact check.
        phase4.validate_canonical_artifact_hashes()
        refreshed = prepare_requests(
            method,
            cases,
            prediction_root=spec.output_path.parents[1],
            log_root=spec.diagnostics_path.parent,
            model_marker_path=DEFAULT_MODEL_MARKER,
        )
        for case, request, _resume in pending:
            fresh = refreshed["requests"][int(case["search_rank"])]
            if request["request_sha256"] != fresh["request_sha256"]:
                raise EvaluationBLLMError(
                    f"Request changed immediately before spend for rank {case['search_rank']}"
                )

        if not pending:
            return materialize(prepared, secret=None, started_at=started_at)

        secret = phase4.require_api_key()
        openai_module, sdk_version = phase4.load_openai_sdk()
        client = phase4.create_openai_client(openai_module, secret)

        def invoke_task(
            task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            case, request, resume = task
            result = phase4.invoke_with_retries(
                client,
                request["payload"],
                max_attempts=max_attempts,
                base_backoff_seconds=base_backoff_seconds,
                secret=secret,
                start_with_fallback=bool(resume["start_with_fallback"]),
                prior_fallback_attempts=int(resume["prior_fallback_attempts"]),
                prior_primary_incomplete_provenance=resume[
                    "prior_primary_incomplete_provenance"
                ],
            )
            return case, request, result

        def commit(
            case: Mapping[str, Any], request: Mapping[str, Any], result: Mapping[str, Any]
        ) -> None:
            rank = int(case["search_rank"])
            if result.get("ok"):
                atomic_json(
                    _success_path(spec, rank),
                    make_api_success_record(
                        method, case, request, result, prepared, sdk_version
                    ),
                    secret=secret,
                )
            else:
                _record_failure(
                    _failure_path(spec, rank),
                    case=case,
                    request=request,
                    result=result,
                    prepared=prepared,
                    method=method,
                    sdk_version=sdk_version,
                    secret=secret,
                )

        if workers == 1:
            for task in pending:
                case, request, result = invoke_task(task)
                commit(case, request, result)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-b-llm") as pool:
                futures: dict[Future[Any], int] = {
                    pool.submit(invoke_task, task): int(task[0]["search_rank"])
                    for task in pending
                }
                for future in as_completed(futures):
                    case, request, result = future.result()
                    commit(case, request, result)
        return materialize(prepared, secret=secret, started_at=started_at)


def build_plan(prepared: Mapping[str, Any]) -> dict[str, Any]:
    reusable, rejected = reusable_prediction_index(
        prepared["spec"].method, prepared["requests"]
    )
    existing = 0
    for case in prepared["cases"]:
        rank = int(case["search_rank"])
        if _load_valid_success(
            _success_path(prepared["spec"], rank),
            case=case,
            request=prepared["requests"][rank],
            method=prepared["spec"].method,
            prepared=prepared,
        ) is not None:
            existing += 1
    return {
        "status": "PLAN_VALIDATED_NO_API_CALL",
        "method": prepared["spec"].method,
        "retained_cases": len(prepared["cases"]),
        "substantive_cases": sum(
            case["review_status"] == "SUBSTANTIVE" for case in prepared["cases"]
        ),
        "abstain_cases": sum(
            case["review_status"] == "ABSTAIN" for case in prepared["cases"]
        ),
        "existing_eval_b_successes": existing,
        "reusable_identical_prior_requests": len(reusable),
        "new_requests_if_executed": len(prepared["cases"]) - existing - len(reusable),
        "rejected_prior_candidates": len(rejected),
        "retained_membership_sha256": prepared["retained_membership_sha256"],
        "demo_bank_id": (prepared.get("demo_metadata") or {}).get("demo_bank_id"),
        "demo_overlap_ranks": prepared["demo_overlap_ranks"],
        "model": prepared["model_marker"]["effective_model_id"],
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("M3", "M4"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--base-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_retained_cases(args.reference, args.sample)
    gate = validate_human_gates(cases, reference_path=args.reference)
    prepared = prepare_requests(
        args.method,
        cases,
        prediction_root=args.prediction_root,
        log_root=args.log_root,
    )
    prepared["human_gate"] = gate
    plan = {**build_plan(prepared), "human_gate": gate}
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    diagnostics = execute(
        prepared,
        workers=args.workers,
        max_attempts=args.max_attempts,
        base_backoff_seconds=args.base_backoff_seconds,
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if diagnostics.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
