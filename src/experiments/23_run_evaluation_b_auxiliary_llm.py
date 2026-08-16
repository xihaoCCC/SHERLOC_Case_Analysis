#!/usr/bin/env python3
"""Run the frozen Evaluation-B auxiliary zero-shot Responses experiment.

The default action is plan-only.  Live execution requires both ``--execute``
and ``--confirm-55-new-requests``.  The runner selects only the 55 frozen
SUBSTANTIVE cases, sends only each exact Fact Summary plus the versioned task
instructions/schema, and never reuses AMP responses.  A validated success for
this exact frozen request is resume-safe and is never resent.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/eval_b_auxiliary_llm_v1.yaml"
DEFAULT_MODEL_MARKER = REPO_ROOT / "outputs/logs/llm/model_access.json"
EXPECTED_CASES = 55
INITIAL_MAX_OUTPUT_TOKENS = 512
FALLBACK_MAX_OUTPUT_TOKENS = 2048
GEO_ORDER = ("Internal", "Transnational")
MULTIPLICITY = ("SINGLE", "MULTIPLE", "UNKNOWN")
CHILD = ("TRUE", "FALSE", "UNKNOWN")
OCG = ("TRUE", "FALSE")


class AuxiliaryLLMError(RuntimeError):
    """Raised before spend when a frozen auxiliary invariant is violated."""


class MaxOutputIncomplete(AuxiliaryLLMError):
    def __init__(self, response: Any) -> None:
        super().__init__("Response was incomplete because max_output_tokens was reached")
        self.response = response


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


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuxiliaryLLMError(f"Cannot read JSON-compatible artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuxiliaryLLMError(f"JSON root is not an object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise AuxiliaryLLMError(f"Cannot read CSV {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
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
        raise AuxiliaryLLMError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Mapping[str, Any], *, secret: str = "") -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if secret and secret in text:
        raise AuxiliaryLLMError("Credential material reached a serialized artifact")
    _atomic_text(path, text)


def atomic_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, secret: str = ""
) -> None:
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    if secret and secret in text:
        raise AuxiliaryLLMError("Credential material reached a serialized artifact")
    _atomic_text(path, text)


def _repo_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AuxiliaryLLMError(f"Config field {field} must be a repository-relative path")
    path = (REPO_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise AuxiliaryLLMError(f"Config field {field} escapes the repository") from exc
    return path


def validate_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AuxiliaryLLMError("Structured output must be an object")
    expected = {
        "geographic_form",
        "multiplicity",
        "child_involvement",
        "organized_criminal_group",
    }
    if set(value) != expected:
        raise AuxiliaryLLMError("Structured output has missing or additional keys")
    raw_geo = value["geographic_form"]
    if not isinstance(raw_geo, list) or any(not isinstance(item, str) for item in raw_geo):
        raise AuxiliaryLLMError("geographic_form must be an array of strings")
    if len(raw_geo) != len(set(raw_geo)):
        raise AuxiliaryLLMError("geographic_form contains a duplicate")
    unknown_geo = set(raw_geo) - set(GEO_ORDER)
    if unknown_geo:
        raise AuxiliaryLLMError(f"geographic_form contains unknown values: {unknown_geo}")
    multiplicity = value["multiplicity"]
    child = value["child_involvement"]
    ocg = value["organized_criminal_group"]
    if multiplicity not in MULTIPLICITY:
        raise AuxiliaryLLMError("Invalid multiplicity output")
    if child not in CHILD:
        raise AuxiliaryLLMError("Invalid child_involvement output")
    if ocg not in OCG:
        raise AuxiliaryLLMError("Invalid organized_criminal_group output")
    selected = set(raw_geo)
    return {
        "geographic_form": [item for item in GEO_ORDER if item in selected],
        "multiplicity": multiplicity,
        "child_involvement": child,
        "organized_criminal_group": ocg,
    }


def validate_schema(schema: Mapping[str, Any]) -> None:
    expected_keys = [
        "geographic_form",
        "multiplicity",
        "child_involvement",
        "organized_criminal_group",
    ]
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or schema.get("required") != expected_keys
        or list((schema.get("properties") or {}).keys()) != expected_keys
    ):
        raise AuxiliaryLLMError("Auxiliary strict schema root changed")
    props = schema["properties"]
    if props["geographic_form"] != {
        "type": "array",
        "items": {"type": "string", "enum": list(GEO_ORDER)},
        "maxItems": 2,
    }:
        raise AuxiliaryLLMError("Geographic Form schema changed")
    for field, allowed in (
        ("multiplicity", MULTIPLICITY),
        ("child_involvement", CHILD),
        ("organized_criminal_group", OCG),
    ):
        if props[field] != {"type": "string", "enum": list(allowed)}:
            raise AuxiliaryLLMError(f"{field} schema changed")


def load_contract(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("status") != "FROZEN_PRE_EXECUTION":
        raise AuxiliaryLLMError("Auxiliary config is not frozen pre-execution")
    if config.get("config_version") != "1.0.0":
        raise AuxiliaryLLMError("Unexpected auxiliary config version")
    membership = config.get("membership") or {}
    method = config.get("method") or {}
    api = config.get("api_request") or {}
    structured = config.get("structured_output") or {}
    input_contract = config.get("input_contract") or {}
    if membership.get("expected_case_count") != EXPECTED_CASES:
        raise AuxiliaryLLMError("Frozen auxiliary membership must contain 55 cases")
    if method != {
        "method_id": "AUX_LLM_ZERO_SHOT",
        "prompt_version": "eval-b-auxiliary-zero-shot-v1",
        "demonstration_count": 0,
        "training_performed": False,
        "case_specific_tuning": False,
    }:
        raise AuxiliaryLLMError("Auxiliary method is no longer the frozen zero-shot method")
    expected_api = {
        "provider": "OpenAI",
        "endpoint": "Responses API",
        "model": "gpt-5.6-luna",
        "one_case_per_request": True,
        "store": False,
        "reasoning": {"effort": "low"},
        "text_verbosity": "low",
        "initial_max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
        "fallback_max_output_tokens": FALLBACK_MAX_OUTPUT_TOKENS,
        "fallback_trigger": (
            "response.status == incomplete and incomplete_details.reason == max_output_tokens"
        ),
        "max_fallback_attempts_per_case": 2,
        "max_transient_attempts": 4,
        "base_backoff_seconds": 2.0,
        "max_backoff_seconds": 60.0,
        "tools": "NONE",
        "previous_response_id": "NONE",
        "temperature": "OMITTED",
        "top_p": "OMITTED",
        "credential_source": "OPENAI_API_KEY_ENVIRONMENT_ONLY",
    }
    if api != expected_api:
        raise AuxiliaryLLMError("Frozen auxiliary API settings changed")
    if input_contract != {
        "case_specific_input": "exact English Fact Summary only",
        "human_reference_labels_sent": False,
        "sherloc_silver_labels_sent": False,
        "prior_model_predictions_sent": False,
        "case_title_sent": False,
        "jurisdiction_sent": False,
        "canonical_url_sent": False,
    }:
        raise AuxiliaryLLMError("Auxiliary input contract changed")
    prompt_path = _repo_path(config["prompt"]["path"], field="prompt.path")
    schema_path = _repo_path(structured["schema_path"], field="schema_path")
    if sha256_file(prompt_path) != config["prompt"].get("sha256"):
        raise AuxiliaryLLMError("Frozen auxiliary prompt hash changed")
    if sha256_file(schema_path) != structured.get("schema_file_sha256"):
        raise AuxiliaryLLMError("Frozen auxiliary schema hash changed")
    if (
        structured.get("format_type") != "json_schema"
        or structured.get("schema_name") != "sherloc_eval_b_auxiliary_v1"
        or structured.get("strict") is not True
        or structured.get("geographic_form_order") != list(GEO_ORDER)
    ):
        raise AuxiliaryLLMError("Structured-output configuration changed")
    schema = load_json(schema_path)
    validate_schema(schema)
    prompt = prompt_path.read_text(encoding="utf-8")
    if "FROZEN_PRE_EXECUTION" not in prompt or "zero-shot" not in prompt.lower():
        raise AuxiliaryLLMError("Prompt lacks its frozen zero-shot marker")
    return {
        "config": config,
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "prompt": prompt,
        "prompt_path": prompt_path,
        "prompt_sha256": sha256_file(prompt_path),
        "schema": schema,
        "schema_path": schema_path,
        "schema_sha256": sha256_file(schema_path),
    }


def load_cases(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = contract["config"]
    membership = config["membership"]
    reference_path = _repo_path(membership["reference_path"], field="reference_path")
    manifest_path = _repo_path(
        membership["membership_manifest_path"], field="membership_manifest_path"
    )
    if sha256_file(reference_path) != membership["reference_sha256"]:
        raise AuxiliaryLLMError("Human-grounded reference hash changed")
    if sha256_file(manifest_path) != membership["membership_manifest_sha256"]:
        raise AuxiliaryLLMError("Evaluation-B membership manifest hash changed")
    manifest = load_json(manifest_path)
    if int(manifest.get("retained_n") or -1) != 61:
        raise AuxiliaryLLMError("Frozen Evaluation-B retained membership is not 61")
    rows = load_csv(reference_path)
    if len(rows) != 61:
        raise AuxiliaryLLMError("Human-grounded reference must contain 61 retained rows")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for row in rows:
        if str(row.get("review_status") or "").strip().upper() != "SUBSTANTIVE":
            continue
        if str(row.get("substantive_amp_evaluable") or "").strip() != "1":
            raise AuxiliaryLLMError("SUBSTANTIVE case lacks its frozen evaluability flag")
        case_id = str(row.get("reliability_case_id") or "").strip()
        fact_summary = str(row.get("fact_summary") or "")
        canonical_url = str(row.get("canonical_url") or "").strip()
        jurisdiction = str(row.get("jurisdiction") or "").strip()
        try:
            rank = int(str(row.get("search_rank") or ""))
        except ValueError as exc:
            raise AuxiliaryLLMError(f"Invalid search rank for {case_id}") from exc
        if (
            not case_id
            or case_id in seen_ids
            or rank <= 0
            or rank in seen_ranks
            or not fact_summary.strip()
            or not canonical_url
            or not jurisdiction
        ):
            raise AuxiliaryLLMError(f"Invalid or duplicated substantive case: {case_id}")
        # Deliberately return no human or silver label fields.  Request assembly
        # cannot accidentally receive a case-specific reference label.
        cases.append(
            {
                "reliability_case_id": case_id,
                "search_rank": rank,
                "canonical_url": canonical_url,
                "jurisdiction": jurisdiction,
                "fact_summary": fact_summary,
                "input_sha256": sha256_text(fact_summary),
            }
        )
        seen_ids.add(case_id)
        seen_ranks.add(rank)
    cases.sort(key=lambda row: int(row["search_rank"]))
    if len(cases) != EXPECTED_CASES:
        raise AuxiliaryLLMError(f"Expected 55 substantive cases, observed {len(cases)}")
    return cases


def membership_sha256(cases: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        canonical_json(
            [
                {
                    "reliability_case_id": row["reliability_case_id"],
                    "search_rank": int(row["search_rank"]),
                    "canonical_url": row["canonical_url"],
                    "input_sha256": row["input_sha256"],
                }
                for row in cases
            ]
        )
    )


def build_payload(case: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    config = contract["config"]
    api = config["api_request"]
    target = {
        "role": "user",
        "content": "Analyze only this supplied Fact Summary:\n"
        + canonical_json({"fact_summary": case["fact_summary"]}),
    }
    payload = {
        "model": api["model"],
        "input": [
            {"role": "developer", "content": contract["prompt"]},
            target,
        ],
        "reasoning": {"effort": "low"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": config["structured_output"]["schema_name"],
                "strict": True,
                "schema": contract["schema"],
            },
        },
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
        "store": False,
    }
    validate_payload(payload, case=case, contract=contract)
    return payload


def validate_payload(
    payload: Mapping[str, Any], *, case: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    expected = {
        "model": "gpt-5.6-luna",
        "input": [
            {"role": "developer", "content": contract["prompt"]},
            {
                "role": "user",
                "content": "Analyze only this supplied Fact Summary:\n"
                + canonical_json({"fact_summary": case["fact_summary"]}),
            },
        ],
        "reasoning": {"effort": "low"},
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "sherloc_eval_b_auxiliary_v1",
                "strict": True,
                "schema": contract["schema"],
            },
        },
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if dict(payload) != expected:
        raise AuxiliaryLLMError("Request payload differs from the frozen exact contract")
    serialized = canonical_json(payload)
    forbidden_keys = (
        "acts_human",
        "means_human",
        "purpose_human",
        "silver_reference",
        "human_reference",
        "predicted_labels",
        "canonical_url",
        "jurisdiction",
        "case_title",
        "search_rank",
        "reliability_case_id",
    )
    if any(f'"{key}"' in serialized for key in forbidden_keys):
        raise AuxiliaryLLMError("Request payload contains forbidden case/reference fields")


def prepare(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract = load_contract(config_path)
    cases = load_cases(contract)
    requests: dict[int, dict[str, Any]] = {}
    for case in cases:
        payload = build_payload(case, contract)
        requests[int(case["search_rank"])] = {
            "payload": payload,
            "request_sha256": sha256_text(canonical_json(payload)),
        }
    return {
        "contract": contract,
        "cases": cases,
        "requests": requests,
        "membership_sha256": membership_sha256(cases),
    }


def validate_model_marker(path: Path = DEFAULT_MODEL_MARKER) -> dict[str, Any]:
    marker = load_json(path)
    if (
        marker.get("status") != "MODEL_ACCESS_CONFIRMED"
        or marker.get("requested_model_id") != "gpt-5.6-luna"
        or not str(marker.get("effective_model_id") or "").startswith("gpt-5.6-luna")
    ):
        raise AuxiliaryLLMError("Existing model-access marker does not confirm gpt-5.6-luna")
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "checked_at": marker.get("checked_at"),
        "requested_model_id": marker.get("requested_model_id"),
        "effective_model_id": marker.get("effective_model_id"),
        "sdk_version": marker.get("sdk_version"),
    }


def pre_execution_freeze(prepared: Mapping[str, Any]) -> dict[str, Any]:
    contract = prepared["contract"]
    config = contract["config"]
    marker = validate_model_marker()
    request_index = [
        {
            "reliability_case_id": case["reliability_case_id"],
            "search_rank": int(case["search_rank"]),
            "input_sha256": case["input_sha256"],
            "request_sha256": prepared["requests"][int(case["search_rank"])][
                "request_sha256"
            ],
        }
        for case in prepared["cases"]
    ]
    return {
        "schema_version": "sherloc-eval-b-auxiliary-pre-execution-freeze-v1",
        "status": "FROZEN_BEFORE_MODEL_PERFORMANCE",
        "frozen_at": utc_now(),
        "expected_new_prediction_cases": EXPECTED_CASES,
        "membership_sha256": prepared["membership_sha256"],
        "request_index_sha256": sha256_text(canonical_json(request_index)),
        "request_index": request_index,
        "artifacts": {
            "config": {
                "path": display_path(contract["config_path"]),
                "sha256": contract["config_sha256"],
            },
            "prompt": {
                "path": display_path(contract["prompt_path"]),
                "sha256": contract["prompt_sha256"],
            },
            "schema": {
                "path": display_path(contract["schema_path"]),
                "sha256": contract["schema_sha256"],
            },
            "human_reference": {
                "path": config["membership"]["reference_path"],
                "sha256": config["membership"]["reference_sha256"],
            },
            "membership_manifest": {
                "path": config["membership"]["membership_manifest_path"],
                "sha256": config["membership"]["membership_manifest_sha256"],
            },
            "runner": {
                "path": display_path(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "model_access_marker": marker,
        "model": "gpt-5.6-luna",
        "demonstration_count": 0,
        "store": False,
        "human_reference_labels_sent": False,
        "silver_reference_labels_sent": False,
        "prior_predictions_sent": False,
    }


def write_or_validate_freeze(prepared: Mapping[str, Any]) -> dict[str, Any]:
    output = _repo_path(
        prepared["contract"]["config"]["outputs"]["pre_execution_freeze"],
        field="outputs.pre_execution_freeze",
    )
    proposed = pre_execution_freeze(prepared)
    if output.is_file():
        existing = load_json(output)
        comparable_existing = dict(existing)
        comparable_proposed = dict(proposed)
        comparable_existing.pop("frozen_at", None)
        comparable_proposed.pop("frozen_at", None)
        if comparable_existing != comparable_proposed:
            raise AuxiliaryLLMError("Existing auxiliary pre-execution freeze has drifted")
        return existing
    atomic_json(output, proposed)
    return proposed


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_primitive(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _safe_primitive(dump(exclude_none=True))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _safe_primitive(to_dict())
    return str(value)


def _incomplete_details(response: Any) -> dict[str, Any] | None:
    value = _get(response, "incomplete_details")
    if value is None:
        return None
    primitive = _safe_primitive(value)
    return dict(primitive) if isinstance(primitive, Mapping) else None


def _usage(response: Any) -> dict[str, Any] | None:
    value = _get(response, "usage")
    if value is None:
        return None
    primitive = _safe_primitive(value)
    return dict(primitive) if isinstance(primitive, Mapping) else {"value": primitive}


def parse_response(response: Any) -> dict[str, Any]:
    status = _get(response, "status")
    details = _incomplete_details(response)
    if status not in (None, "completed"):
        if status == "incomplete" and details and details.get("reason") == "max_output_tokens":
            raise MaxOutputIncomplete(response)
        raise AuxiliaryLLMError(f"Response status is {status!r}; details={details}")
    output = _get(response, "output", []) or []
    for item in output:
        for content in _get(item, "content", []) or []:
            if _get(content, "type") == "refusal":
                raise AuxiliaryLLMError("The API returned a model refusal")
    output_text = _get(response, "output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise AuxiliaryLLMError("API response contains no structured output text")
    try:
        raw = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise AuxiliaryLLMError("Structured output text is not valid JSON") from exc
    return {
        "raw_structured_response_text": output_text,
        "raw_structured_response": raw,
        "validated_prediction": validate_output(raw),
    }


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _transient(exc: BaseException) -> bool:
    status = _status_code(exc)
    return status in {408, 409, 425, 429} or (status is not None and status >= 500) or type(
        exc
    ).__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        try:
            return max(0.0, float(milliseconds) / 1000.0)
        except (TypeError, ValueError):
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            when = parsedate_to_datetime(str(value))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _response_provenance(response: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "response_id": str(_get(response, "id") or "") or None,
        "returned_model_id": str(_get(response, "model") or "") or None,
        "response_status": _get(response, "status"),
        "incomplete_details": _incomplete_details(response),
        "token_usage": _usage(response),
        "max_output_tokens": payload["max_output_tokens"],
        "actual_request_sha256": sha256_text(canonical_json(payload)),
    }


def _error_event(
    exc: BaseException,
    payload: Mapping[str, Any],
    *,
    phase: str,
    call_number: int,
    secret: str,
) -> dict[str, Any]:
    message = str(exc).replace(secret, "[REDACTED]")[:1000]
    event = {
        "timestamp": utc_now(),
        "phase": phase,
        "call_number": call_number,
        "max_output_tokens": payload["max_output_tokens"],
        "actual_request_sha256": sha256_text(canonical_json(payload)),
        "error_type": type(exc).__name__,
        "http_status": _status_code(exc),
        "transient": _transient(exc),
        "message": message,
    }
    if isinstance(exc, MaxOutputIncomplete):
        event.update(_response_provenance(exc.response, payload))
        event["technical_fallback_trigger"] = (
            phase == "INITIAL_512"
            and event.get("response_status") == "incomplete"
            and (event.get("incomplete_details") or {}).get("reason")
            == "max_output_tokens"
        )
    return event


def fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("max_output_tokens") != INITIAL_MAX_OUTPUT_TOKENS:
        raise AuxiliaryLLMError("Fallback requires the exact 512-token base payload")
    result = dict(payload)
    result["max_output_tokens"] = FALLBACK_MAX_OUTPUT_TOKENS
    left, right = dict(payload), dict(result)
    left.pop("max_output_tokens")
    right.pop("max_output_tokens")
    if left != right:
        raise AuxiliaryLLMError("Fallback changed a semantic request setting")
    return result


def invoke_with_retries(
    client: Any,
    payload: Mapping[str, Any],
    *,
    secret: str,
    prior_failure: Mapping[str, Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    request_sha = sha256_text(canonical_json(payload))
    attempts: list[dict[str, Any]] = []
    initial_trigger: dict[str, Any] | None = None
    prior_fallback_calls = 0
    if prior_failure:
        if prior_failure.get("request_sha256") != request_sha:
            raise AuxiliaryLLMError("Persisted failure belongs to a different request")
        initial_trigger = prior_failure.get("initial_incomplete_provenance")
        prior_fallback_calls = int(prior_failure.get("fallback_calls") or 0)
        if initial_trigger is not None and not (
            isinstance(initial_trigger, Mapping)
            and initial_trigger.get("response_status") == "incomplete"
            and (initial_trigger.get("incomplete_details") or {}).get("reason")
            == "max_output_tokens"
            and initial_trigger.get("max_output_tokens") == INITIAL_MAX_OUTPUT_TOKENS
            and initial_trigger.get("actual_request_sha256") == request_sha
        ):
            raise AuxiliaryLLMError("Persisted fallback trigger is invalid")
        if prior_fallback_calls < 0 or prior_fallback_calls > 2:
            raise AuxiliaryLLMError("Persisted fallback count is invalid")
    started = time.perf_counter()
    call_number = 0
    if initial_trigger is None:
        for attempt in range(1, 5):
            call_number += 1
            try:
                response = client.responses.create(**dict(payload))
                parsed = parse_response(response)
                return {
                    "ok": True,
                    "response": response,
                    "parsed": parsed,
                    "attempts": attempts,
                    "request_calls": call_number,
                    "fallback_calls": prior_fallback_calls,
                    "initial_incomplete_provenance": None,
                    "effective_max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
                    "latency_seconds": time.perf_counter() - started,
                }
            except Exception as exc:
                event = _error_event(
                    exc, payload, phase="INITIAL_512", call_number=call_number, secret=secret
                )
                attempts.append(event)
                if isinstance(exc, MaxOutputIncomplete):
                    initial_trigger = {
                        key: event.get(key)
                        for key in (
                            "timestamp",
                            "response_id",
                            "returned_model_id",
                            "response_status",
                            "incomplete_details",
                            "token_usage",
                            "max_output_tokens",
                            "actual_request_sha256",
                        )
                    }
                    break
                if _transient(exc) and attempt < 4:
                    delay = min(60.0, max(2.0 * (2 ** (attempt - 1)), _retry_after(exc) or 0.0))
                    event["sleep_seconds"] = delay
                    sleeper(delay)
                    continue
                return {
                    "ok": False,
                    "error": event,
                    "attempts": attempts,
                    "request_calls": call_number,
                    "fallback_calls": prior_fallback_calls,
                    "initial_incomplete_provenance": None,
                    "latency_seconds": time.perf_counter() - started,
                }
    if initial_trigger is None:
        raise AssertionError("fallback cannot begin without an explicit incomplete trigger")
    if prior_fallback_calls >= 2:
        return {
            "ok": False,
            "error": {"error_type": "FallbackCeilingExhausted"},
            "attempts": attempts,
            "request_calls": call_number,
            "fallback_calls": prior_fallback_calls,
            "initial_incomplete_provenance": initial_trigger,
            "latency_seconds": time.perf_counter() - started,
        }
    expanded = fallback_payload(payload)
    for ordinal in range(prior_fallback_calls + 1, 3):
        call_number += 1
        try:
            response = client.responses.create(**dict(expanded))
            parsed = parse_response(response)
            return {
                "ok": True,
                "response": response,
                "parsed": parsed,
                "attempts": attempts,
                "request_calls": call_number,
                "fallback_calls": ordinal,
                "initial_incomplete_provenance": initial_trigger,
                "effective_max_output_tokens": FALLBACK_MAX_OUTPUT_TOKENS,
                "latency_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            event = _error_event(
                exc, expanded, phase="FALLBACK_2048", call_number=call_number, secret=secret
            )
            attempts.append(event)
            if (isinstance(exc, MaxOutputIncomplete) or _transient(exc)) and ordinal < 2:
                delay = min(60.0, max(2.0 * (2 ** (ordinal - 1)), _retry_after(exc) or 0.0))
                event["sleep_seconds"] = delay
                sleeper(delay)
                continue
            return {
                "ok": False,
                "error": event,
                "attempts": attempts,
                "request_calls": call_number,
                "fallback_calls": ordinal,
                "initial_incomplete_provenance": initial_trigger,
                "latency_seconds": time.perf_counter() - started,
            }
    raise AssertionError("unreachable")


def _state_paths(prepared: Mapping[str, Any], rank: int) -> tuple[Path, Path]:
    config = prepared["contract"]["config"]
    diagnostics = _repo_path(config["outputs"]["diagnostics"], field="outputs.diagnostics")
    state = diagnostics.parent / "state"
    return state / "success" / f"{rank:06d}.json", state / "failures" / f"{rank:06d}.json"


def _valid_success(
    path: Path,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = load_json(path)
    expected = {
        "status": "SUCCESS_VALIDATED",
        "method": "AUX_LLM_ZERO_SHOT",
        "reliability_case_id": case["reliability_case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": case["input_sha256"],
        "request_sha256": request["request_sha256"],
        "membership_sha256": prepared["membership_sha256"],
        "prompt_sha256": prepared["contract"]["prompt_sha256"],
        "schema_sha256": prepared["contract"]["schema_sha256"],
        "config_sha256": prepared["contract"]["config_sha256"],
        "requested_model_id": "gpt-5.6-luna",
        "api_request_issued": True,
        "demonstration_count": 0,
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise AuxiliaryLLMError(f"Stale auxiliary success artifact: {path}")
    raw_text = row.get("raw_structured_response_text")
    try:
        decoded_raw = json.loads(raw_text) if isinstance(raw_text, str) else None
    except json.JSONDecodeError as exc:
        raise AuxiliaryLLMError(f"Invalid raw response text: {path}") from exc
    raw = row.get("raw_structured_response")
    prediction = validate_output(raw)
    if (
        decoded_raw != raw
        or row.get("validated_prediction") != prediction
    ):
        raise AuxiliaryLLMError(f"Noncanonical auxiliary prediction: {path}")
    return row


def _success_record(
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    prepared: Mapping[str, Any],
    sdk_version: str,
    prior_failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    response = result["response"]
    parsed = result["parsed"]
    prior_calls = int((prior_failure or {}).get("request_calls") or 0)
    prior_attempts = list((prior_failure or {}).get("attempts") or [])
    current_attempts = list(result["attempts"])
    return {
        "schema_version": "sherloc-eval-b-auxiliary-prediction-v1",
        "runner_version": VERSION,
        "status": "SUCCESS_VALIDATED",
        "method": "AUX_LLM_ZERO_SHOT",
        "evaluation": "A3_HUMAN_GROUNDED_AUXILIARY",
        "reliability_case_id": case["reliability_case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": case["input_sha256"],
        "request_sha256": request["request_sha256"],
        "membership_sha256": prepared["membership_sha256"],
        "prompt_sha256": prepared["contract"]["prompt_sha256"],
        "schema_sha256": prepared["contract"]["schema_sha256"],
        "config_sha256": prepared["contract"]["config_sha256"],
        "validated_prediction": parsed["validated_prediction"],
        "raw_structured_response": parsed["raw_structured_response"],
        "raw_structured_response_text": parsed["raw_structured_response_text"],
        "requested_model_id": "gpt-5.6-luna",
        "returned_model_id": str(_get(response, "model") or "") or None,
        "response_id": str(_get(response, "id") or "") or None,
        "token_usage": _usage(response),
        "request_calls_this_invocation": result["request_calls"],
        "prior_request_calls": prior_calls,
        "request_calls_cumulative": prior_calls + int(result["request_calls"]),
        "fallback_calls_cumulative": result["fallback_calls"],
        "initial_incomplete_provenance": result["initial_incomplete_provenance"],
        "effective_max_output_tokens": result["effective_max_output_tokens"],
        "retry_events": prior_attempts + current_attempts,
        "latency_seconds": result["latency_seconds"],
        "sdk_version": sdk_version,
        "execution_timestamp": utc_now(),
        "api_request_issued": True,
        "demonstration_count": 0,
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
    }


def materialize(prepared: Mapping[str, Any], *, secret: str = "") -> dict[str, Any]:
    config = prepared["contract"]["config"]
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    missing: list[int] = []
    for case in prepared["cases"]:
        rank = int(case["search_rank"])
        success_path, failure_path = _state_paths(prepared, rank)
        row = _valid_success(success_path, case, prepared["requests"][rank], prepared)
        if row is not None:
            successes.append(row)
        elif failure_path.is_file():
            failures.append(load_json(failure_path))
        else:
            missing.append(rank)
    successes.sort(key=lambda row: int(row["search_rank"]))
    prediction_path = _repo_path(config["outputs"]["predictions"], field="outputs.predictions")
    diagnostics_path = _repo_path(config["outputs"]["diagnostics"], field="outputs.diagnostics")
    atomic_jsonl(prediction_path, successes, secret=secret)
    diagnostics = {
        "schema_version": "sherloc-eval-b-auxiliary-diagnostics-v1",
        "status": "COMPLETE" if len(successes) == EXPECTED_CASES else "INCOMPLETE",
        "expected_cases": EXPECTED_CASES,
        "successful_predictions": len(successes),
        "unresolved_failures": len(failures),
        "missing_unattempted": len(missing),
        "failure_ranks": sorted(int(row["search_rank"]) for row in failures),
        "missing_ranks": missing,
        "prediction_path": display_path(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "membership_sha256": prepared["membership_sha256"],
        "config_sha256": prepared["contract"]["config_sha256"],
        "prompt_sha256": prepared["contract"]["prompt_sha256"],
        "schema_sha256": prepared["contract"]["schema_sha256"],
        "model": "gpt-5.6-luna",
        "zero_shot": True,
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
        "api_success_records": sum(row.get("api_request_issued") is True for row in successes),
        "total_request_calls": sum(int(row.get("request_calls_cumulative") or 0) for row in successes),
        "total_retry_calls": sum(
            max(0, int(row.get("request_calls_cumulative") or 0) - 1)
            for row in successes
        ),
        "retry_event_count": sum(len(row.get("retry_events") or []) for row in successes),
        "fallback_cases": sum(int(row.get("fallback_calls_cumulative") or 0) > 0 for row in successes),
        "fallback_calls": sum(int(row.get("fallback_calls_cumulative") or 0) for row in successes),
        "completed_at": utc_now(),
    }
    atomic_json(diagnostics_path, diagnostics, secret=secret)
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
            raise AuxiliaryLLMError("Another auxiliary execution holds the run lock") from exc
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def execute(prepared: Mapping[str, Any], *, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 4:
        raise AuxiliaryLLMError("workers must be between 1 and 4")
    freeze = write_or_validate_freeze(prepared)
    config = prepared["contract"]["config"]
    freeze_path = _repo_path(config["outputs"]["pre_execution_freeze"], field="freeze")
    if freeze.get("status") != "FROZEN_BEFORE_MODEL_PERFORMANCE":
        raise AuxiliaryLLMError("Pre-execution freeze gate did not pass")
    # Rebuild all requests immediately before reading a credential.
    refreshed = prepare(prepared["contract"]["config_path"])
    if {
        rank: item["request_sha256"] for rank, item in refreshed["requests"].items()
    } != {rank: item["request_sha256"] for rank, item in prepared["requests"].items()}:
        raise AuxiliaryLLMError("Requests changed immediately before execution")
    secret = os.environ.get("OPENAI_API_KEY", "")
    if not secret.strip():
        raise AuxiliaryLLMError("OPENAI_API_KEY is absent from the process environment")
    try:
        import openai  # type: ignore
    except ImportError as exc:
        raise AuxiliaryLLMError("The official OpenAI Python SDK is required") from exc
    sdk_version = str(getattr(openai, "__version__", "UNKNOWN"))
    client = openai.OpenAI(api_key=secret, max_retries=0)
    diagnostics_path = _repo_path(config["outputs"]["diagnostics"], field="diagnostics")
    lock_path = diagnostics_path.parent / "state/.run.guard"
    pending: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = []
    with RunLock(lock_path):
        for case in prepared["cases"]:
            rank = int(case["search_rank"])
            request = prepared["requests"][rank]
            success_path, failure_path = _state_paths(prepared, rank)
            if _valid_success(success_path, case, request, prepared) is not None:
                continue
            failure = load_json(failure_path) if failure_path.is_file() else None
            pending.append((case, request, failure))

        def invoke(task: tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]) -> tuple[Any, ...]:
            case, request, prior = task
            result = invoke_with_retries(
                client, request["payload"], secret=secret, prior_failure=prior
            )
            return case, request, result, prior

        def commit(
            case: Mapping[str, Any],
            request: Mapping[str, Any],
            result: Mapping[str, Any],
            prior_failure: Mapping[str, Any] | None,
        ) -> None:
            rank = int(case["search_rank"])
            success_path, failure_path = _state_paths(prepared, rank)
            if result.get("ok"):
                atomic_json(
                    success_path,
                    _success_record(
                        case,
                        request,
                        result,
                        prepared,
                        sdk_version,
                        prior_failure,
                    ),
                    secret=secret,
                )
                if failure_path.is_file():
                    # Preserve history inside the success record; do not delete the
                    # failure file.  It is ignored once a validated success exists.
                    pass
                print(canonical_json({"status": "SUCCESS", "search_rank": rank}), flush=True)
            else:
                prior = dict(prior_failure or {})
                prior_calls = int(prior.get("request_calls") or 0)
                failure = {
                    "schema_version": "sherloc-eval-b-auxiliary-failure-v1",
                    "status": "FAILED_NO_PREDICTION",
                    "reliability_case_id": case["reliability_case_id"],
                    "search_rank": rank,
                    "request_sha256": request["request_sha256"],
                    "input_sha256": case["input_sha256"],
                    "initial_incomplete_provenance": result.get("initial_incomplete_provenance"),
                    "fallback_calls": result.get("fallback_calls", prior.get("fallback_calls", 0)),
                    "request_calls": prior_calls + int(result.get("request_calls") or 0),
                    "last_error": result.get("error"),
                    "attempts": list(prior.get("attempts") or []) + list(result.get("attempts") or []),
                    "updated_at": utc_now(),
                }
                atomic_json(failure_path, failure, secret=secret)
                print(canonical_json({"status": "FAILED", "search_rank": rank}), flush=True)

        if workers == 1:
            for task in pending:
                case, request, result, prior = invoke(task)
                commit(case, request, result, prior)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="eval-b-aux") as pool:
                futures: dict[Future[Any], int] = {
                    pool.submit(invoke, task): int(task[0]["search_rank"]) for task in pending
                }
                for future in as_completed(futures):
                    case, request, result, prior = future.result()
                    commit(case, request, result, prior)
    diagnostics = materialize(prepared, secret=secret)
    diagnostics["pre_execution_freeze_sha256"] = sha256_file(freeze_path)
    atomic_json(diagnostics_path, diagnostics, secret=secret)
    return diagnostics


def build_plan(prepared: Mapping[str, Any]) -> dict[str, Any]:
    config = prepared["contract"]["config"]
    existing = 0
    for case in prepared["cases"]:
        rank = int(case["search_rank"])
        success_path, _ = _state_paths(prepared, rank)
        if _valid_success(success_path, case, prepared["requests"][rank], prepared) is not None:
            existing += 1
    return {
        "status": "PLAN_ONLY_NO_API_REQUEST",
        "model": "gpt-5.6-luna",
        "zero_shot": True,
        "expected_cases": EXPECTED_CASES,
        "existing_validated_successes": existing,
        "new_cases_if_executed": EXPECTED_CASES - existing,
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
        "membership_sha256": prepared["membership_sha256"],
        "config_sha256": prepared["contract"]["config_sha256"],
        "prompt_sha256": prepared["contract"]["prompt_sha256"],
        "schema_sha256": prepared["contract"]["schema_sha256"],
        "pre_execution_freeze_path": config["outputs"]["pre_execution_freeze"],
        "prediction_path": config["outputs"]["predictions"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-55-new-requests", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepared = prepare(args.config)
        plan = build_plan(prepared)
        if not args.execute:
            freeze = write_or_validate_freeze(prepared)
            plan["pre_execution_freeze_sha256"] = sha256_file(
                _repo_path(
                    prepared["contract"]["config"]["outputs"]["pre_execution_freeze"],
                    field="pre_execution_freeze",
                )
            )
            plan["freeze_status"] = freeze["status"]
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not args.confirm_55_new_requests:
            raise AuxiliaryLLMError(
                "Live execution requires --confirm-55-new-requests"
            )
        result = execute(prepared, workers=args.workers)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "COMPLETE" else 2
    except AuxiliaryLLMError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
