#!/usr/bin/env python3
"""Build reproducible AMP-only M3/M4 Responses API payloads without sending.

This module is deliberately side-effect free.  It never imports an API client,
reads an API key, sends a request, or writes predictions.  M4 fails closed
unless the caller explicitly supplies exactly six ordered, unique,
human-approved and frozen demonstrations.

Expected target-case mapping::

    {
        "case_id": "research-stable-case-id",
        "search_rank": 123,
        "canonical_url": "https://...",
        "fact_summary": "Exact English Fact Summary"
    }

Each M4 demonstration must additionally contain ``demo_id``, ``demo_order``,
``jurisdiction``, ``output``, ``human_approved=True``, ``frozen=True``, and a
nonempty ``approval_record``.  ``output`` must match the strict AMP-only schema.
No implicit loading from a candidate or review CSV is supported: that would
bypass the required human freeze.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config/experiments/llm_extraction_amp_v2.yaml"
DEFAULT_M3_PROMPT_PATH = REPO_ROOT / "prompts/m3_zero_shot_amp_v2.md"
DEFAULT_M4_PROMPT_PATH = REPO_ROOT / "prompts/m4_six_shot_amp_v2.md"

SHARED_BEGIN = "<!-- SHERLOC_SHARED_INSTRUCTIONS_V2_BEGIN -->"
SHARED_END = "<!-- SHERLOC_SHARED_INSTRUCTIONS_V2_END -->"

ACT_IDS = (
    "ACT_RECRUITMENT",
    "ACT_TRANSPORTATION",
    "ACT_TRANSFER",
    "ACT_HARBOURING",
    "ACT_RECEIPT",
)
MEANS_IDS = (
    "MEANS_THREAT_FORCE_OR_COERCION",
    "MEANS_ABDUCTION",
    "MEANS_FRAUD",
    "MEANS_DECEPTION",
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
)
PURPOSE_IDS = (
    "PURPOSE_SEXUAL_EXPLOITATION",
    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
    "PURPOSE_SERVITUDE",
    "PURPOSE_REMOVAL_OF_ORGANS",
    "PURPOSE_OTHER",
)

REQUIRED_CASE_FIELDS = (
    "case_id",
    "search_rank",
    "canonical_url",
    "fact_summary",
)
REQUIRED_DEMO_FIELDS = (
    "demo_id",
    "demo_order",
    "search_rank",
    "canonical_url",
    "jurisdiction",
    "fact_summary",
    "output",
    "human_approved",
    "frozen",
    "approval_record",
)
PLACEHOLDER_BANK_TERMS = (
    "UNFROZEN",
    "NOT_FROZEN",
    "PENDING",
    "PROPOSAL",
    "DRAFT",
    "PLACEHOLDER",
    "TBD",
)


class RequestBuildError(ValueError):
    """Raised when a prompt, config, case, or demo freeze invariant fails."""


def canonical_json(value: Any) -> str:
    """Return the UTF-8 hash representation used throughout this module."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RequestBuildError(f"Value is not canonical-JSON serializable: {exc}") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json_yaml(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML 1.2 config without a PyYAML dependency."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RequestBuildError(f"Cannot load experiment config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RequestBuildError(f"Experiment config root must be an object: {path}")
    return value


def extract_marked_instruction_block(prompt_text: str) -> tuple[str, str]:
    """Return the full marked block and its inner developer-message content."""

    if prompt_text.count(SHARED_BEGIN) != 1 or prompt_text.count(SHARED_END) != 1:
        raise RequestBuildError("Prompt must contain exactly one shared instruction block")
    start = prompt_text.index(SHARED_BEGIN)
    end = prompt_text.index(SHARED_END, start) + len(SHARED_END)
    if end <= start:
        raise RequestBuildError("Shared instruction markers are out of order")
    full_block = prompt_text[start:end]
    inner = full_block[len(SHARED_BEGIN) : -len(SHARED_END)]
    if inner.startswith("\n"):
        inner = inner[1:]
    if inner.endswith("\n"):
        inner = inner[:-1]
    if not inner.strip():
        raise RequestBuildError("Shared instruction block is empty")
    return full_block, inner


def load_shared_prompt_contract(
    m3_prompt_path: Path = DEFAULT_M3_PROMPT_PATH,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT_PATH,
) -> dict[str, str]:
    """Load M3/M4 and enforce a byte-identical marked instruction block."""

    try:
        m3_text = m3_prompt_path.read_text(encoding="utf-8")
        m4_text = m4_prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RequestBuildError(f"Cannot load prompt specification: {exc}") from exc
    m3_block, m3_inner = extract_marked_instruction_block(m3_text)
    m4_block, m4_inner = extract_marked_instruction_block(m4_text)
    if m3_block.encode("utf-8") != m4_block.encode("utf-8"):
        raise RequestBuildError("M3 and M4 marked instruction blocks are not byte-identical")
    if m3_inner != m4_inner:  # Defensive; the byte comparison already implies this.
        raise RequestBuildError("M3 and M4 developer instructions differ")
    return {
        "marked_block": m3_block,
        "developer_instruction": m3_inner,
        "marked_block_sha256": sha256_text(m3_block),
    }


def _contains_key(value: Any, forbidden_key: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden_key in value or any(
            _contains_key(item, forbidden_key) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, forbidden_key) for item in value)
    return False


def _validate_config(config: Mapping[str, Any]) -> None:
    try:
        ontology = config["ontology"]
        structured = config["structured_output"]
        schema = structured["schema"]
        properties = schema["properties"]
        api = config["api_request"]
        methods = config["methods"]
    except (KeyError, TypeError) as exc:
        raise RequestBuildError(f"LLM config is missing a required field: {exc}") from exc

    expected = {
        "act_ids": list(ACT_IDS),
        "means_ids": list(MEANS_IDS),
        "purpose_ids": list(PURPOSE_IDS),
    }
    for name, ids in expected.items():
        if ontology.get(name) != ids:
            raise RequestBuildError(f"Frozen ontology mismatch in config field {name}")

    if structured.get("format_type") != "json_schema" or structured.get("strict") is not True:
        raise RequestBuildError("Structured Outputs must use strict json_schema format")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise RequestBuildError("Structured-output root must be a closed object")
    if schema.get("required") != ["acts", "means", "purposes"]:
        raise RequestBuildError("Structured-output root fields or ordering changed")
    if set(properties) != {"acts", "means", "purposes"}:
        raise RequestBuildError("Structured-output root contains a non-AMP field")
    schema_expectations = (
        ("acts", ACT_IDS),
        ("means", MEANS_IDS),
        ("purposes", PURPOSE_IDS),
    )
    for field, ids in schema_expectations:
        item = properties.get(field, {})
        if item.get("type") != "array":
            raise RequestBuildError(f"{field} must be an array")
        if item.get("items", {}).get("enum") != list(ids):
            raise RequestBuildError(f"{field} enum no longer matches the frozen ontology")
        if item.get("maxItems") != len(ids):
            raise RequestBuildError(f"{field} maxItems must equal its ontology size")
    if _contains_key(schema, "uniqueItems"):
        raise RequestBuildError(
            "uniqueItems is not part of the documented Structured Outputs array subset"
        )

    if api.get("provider") != "OpenAI" or api.get("endpoint") != "Responses API":
        raise RequestBuildError("Only the frozen OpenAI Responses API contract is supported")
    if api.get("model") != "gpt-5.6-luna":
        raise RequestBuildError("Frozen API model must be gpt-5.6-luna")
    if api.get("reasoning") != {"effort": "low"}:
        raise RequestBuildError("Frozen reasoning configuration must contain only effort=low")
    if api.get("text_verbosity") != "low" or api.get("max_output_tokens") != 512:
        raise RequestBuildError("Frozen text verbosity or output-token limit changed")
    if api.get("store") is not False or api.get("one_case_per_request") is not True:
        raise RequestBuildError("Requests must be independent and store=false")
    if methods.get("M3", {}).get("demonstration_count") != 0:
        raise RequestBuildError("M3 must contain zero demonstrations")
    if methods.get("M4", {}).get("demonstration_count") != 6:
        raise RequestBuildError("M4 must contain exactly six demonstrations")


def load_contract(
    config_path: Path = DEFAULT_CONFIG_PATH,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT_PATH,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT_PATH,
) -> dict[str, Any]:
    config = _load_json_yaml(config_path)
    _validate_config(config)
    prompt = load_shared_prompt_contract(m3_prompt_path, m4_prompt_path)
    for method, path in (("M3", m3_prompt_path), ("M4", m4_prompt_path)):
        observed = sha256_text(path.read_text(encoding="utf-8"))
        expected = config["methods"][method].get("prompt_sha256")
        if expected != observed:
            raise RequestBuildError(f"{method} full prompt hash does not match frozen config")
    if config["shared_task"].get("shared_marked_block_sha256") != prompt[
        "marked_block_sha256"
    ]:
        raise RequestBuildError("Shared instruction hash does not match frozen config")
    schema_hash = sha256_text(canonical_json(config["structured_output"]["schema"]))
    if config["structured_output"].get("schema_sha256") != schema_hash:
        raise RequestBuildError("Structured-output schema hash does not match frozen config")
    return {
        "config": config,
        "config_sha256": sha256_text(config_path.read_text(encoding="utf-8")),
        **prompt,
    }


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RequestBuildError(f"{field} must be a nonempty string")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RequestBuildError(f"{field} must be a positive integer")
    return value


def _validate_case(case: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
    if missing:
        raise RequestBuildError(f"Target case is missing fields: {', '.join(missing)}")
    return {
        "case_id": _nonempty_string(case["case_id"], "case_id"),
        "search_rank": _positive_int(case["search_rank"], "search_rank"),
        "canonical_url": _nonempty_string(case["canonical_url"], "canonical_url"),
        # Preserve the original response string byte-for-byte; validation strips
        # only to decide whether content exists.
        "fact_summary": _nonempty_string(case["fact_summary"], "fact_summary"),
    }


def _validated_canonical_labels(
    value: Any, allowed: Sequence[str], field: str
) -> list[str]:
    """Validate semantic membership, then canonicalize an unordered label set.

    Multi-label array order has no semantic meaning.  The versioned technical
    amendment therefore preserves strict rejection of non-strings, duplicates,
    and unknown labels while ordering every valid set by the frozen ontology
    before persistence or scoring.
    """

    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RequestBuildError(f"{field} must be an array of strings")
    if len(value) != len(set(value)):
        raise RequestBuildError(f"{field} contains duplicate labels")
    unknown = [item for item in value if item not in allowed]
    if unknown:
        raise RequestBuildError(f"{field} contains unknown labels: {unknown}")
    selected = set(value)
    return [item for item in allowed if item in selected]


def validate_structured_output(output: Any) -> dict[str, Any]:
    """Validate and return a schema-ordered copy of one AMP-only output."""

    if not isinstance(output, Mapping):
        raise RequestBuildError("Structured output must be an object")
    expected_keys = {"acts", "means", "purposes"}
    if set(output) != expected_keys:
        raise RequestBuildError(
            f"Structured output keys must be exactly {sorted(expected_keys)}"
        )
    acts = _validated_canonical_labels(output["acts"], ACT_IDS, "acts")
    means = _validated_canonical_labels(output["means"], MEANS_IDS, "means")
    purposes = _validated_canonical_labels(
        output["purposes"], PURPOSE_IDS, "purposes"
    )
    return {
        "acts": acts,
        "means": means,
        "purposes": purposes,
    }


def _validate_demo_bank(
    demos: Sequence[Mapping[str, Any]] | None,
    demo_bank_version: str | None,
    heldout_jurisdictions: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if demos is None or isinstance(demos, (str, bytes)):
        raise RequestBuildError("M4 requires six explicitly supplied demonstration objects")
    if len(demos) != 6:
        raise RequestBuildError(f"M4 requires exactly six demonstrations, got {len(demos)}")
    version = _nonempty_string(demo_bank_version, "demo_bank_version")
    upper_version = version.upper().replace("-", "_").replace(" ", "_")
    if any(term in upper_version for term in PLACEHOLDER_BANK_TERMS):
        raise RequestBuildError("demo_bank_version is provisional rather than frozen")

    heldout = set(heldout_jurisdictions or ())
    normalized: list[dict[str, Any]] = []
    demo_ids: set[str] = set()
    ranks: set[int] = set()
    urls: set[str] = set()
    for expected_order, raw in enumerate(demos, start=1):
        if not isinstance(raw, Mapping):
            raise RequestBuildError(f"Demonstration {expected_order} must be an object")
        missing = [field for field in REQUIRED_DEMO_FIELDS if field not in raw]
        if missing:
            raise RequestBuildError(
                f"Demonstration {expected_order} is missing fields: {', '.join(missing)}"
            )
        if raw["human_approved"] is not True or raw["frozen"] is not True:
            raise RequestBuildError(
                f"Demonstration {expected_order} lacks explicit human approval/freeze"
            )
        order = _positive_int(raw["demo_order"], "demo_order")
        if order != expected_order:
            raise RequestBuildError(
                f"Demonstration order must be exactly 1..6; expected {expected_order}, got {order}"
            )
        demo_id = _nonempty_string(raw["demo_id"], "demo_id")
        rank = _positive_int(raw["search_rank"], "search_rank")
        url = _nonempty_string(raw["canonical_url"], "canonical_url")
        jurisdiction = _nonempty_string(raw["jurisdiction"], "jurisdiction")
        if jurisdiction in heldout:
            raise RequestBuildError(
                f"Demonstration {expected_order} jurisdiction is held out for this fold"
            )
        if demo_id in demo_ids or rank in ranks or url in urls:
            raise RequestBuildError("Demonstration IDs, ranks, and URLs must each be unique")
        demo_ids.add(demo_id)
        ranks.add(rank)
        urls.add(url)
        normalized.append(
            {
                "demo_id": demo_id,
                "demo_order": order,
                "search_rank": rank,
                "canonical_url": url,
                "jurisdiction": jurisdiction,
                "fact_summary": _nonempty_string(raw["fact_summary"], "fact_summary"),
                "output": validate_structured_output(raw["output"]),
                "human_approved": True,
                "frozen": True,
                "approval_record": _nonempty_string(
                    raw["approval_record"], "approval_record"
                ),
            }
        )
    bank_hash = sha256_text(canonical_json({"version": version, "demos": normalized}))
    return normalized, bank_hash


def _summary_message(fact_summary: str) -> dict[str, str]:
    # JSON encoding prevents delimiter ambiguity while preserving the exact
    # supplied string.  The same wrapper is used for targets and demonstrations.
    return {
        "role": "user",
        "content": "Extract this case from the supplied evidence only:\n"
        + canonical_json({"fact_summary": fact_summary}),
    }


def _assistant_demo_message(output: Mapping[str, Any]) -> dict[str, str]:
    # validate_structured_output has already produced schema key order.  Keep it
    # here rather than sort keys so demonstrations mirror API schema ordering.
    content = json.dumps(
        output,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {"role": "assistant", "content": content}


def _base_payload(contract: Mapping[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    config = contract["config"]
    api = config["api_request"]
    structured = config["structured_output"]
    payload: dict[str, Any] = {
        "model": api["model"],
        "input": messages,
        "reasoning": copy.deepcopy(api["reasoning"]),
        "text": {
            "verbosity": api["text_verbosity"],
            "format": {
                "type": structured["format_type"],
                "name": structured["schema_name"],
                "strict": structured["strict"],
                "schema": copy.deepcopy(structured["schema"]),
            },
        },
        "max_output_tokens": api["max_output_tokens"],
        "store": api["store"],
    }
    # No tools, previous_response_id, sampling parameter, identifier, label, or
    # secret is inserted.  Each returned payload is an independent request.
    return payload


def _result(
    *,
    method: str,
    target: Mapping[str, Any],
    contract: Mapping[str, Any],
    payload: Mapping[str, Any],
    demo_bank_version: str | None,
    demo_bank_sha256: str | None,
) -> dict[str, Any]:
    method_config = contract["config"]["methods"][method]
    metadata = {
        "experiment_id": method_config["experiment_id"],
        "case_id": target["case_id"],
        "search_rank": target["search_rank"],
        "canonical_url": target["canonical_url"],
        "model": contract["config"]["api_request"]["model"],
        "prompt_version": method_config["prompt_version"],
        "demo_bank_version": demo_bank_version,
        "input_text_sha256": sha256_text(target["fact_summary"]),
        "shared_instruction_sha256": contract["marked_block_sha256"],
        "schema_sha256": sha256_text(
            canonical_json(contract["config"]["structured_output"]["schema"])
        ),
        "demo_bank_sha256": demo_bank_sha256,
        "config_sha256": contract["config_sha256"],
        "request_payload_sha256": sha256_text(canonical_json(payload)),
    }
    return {"metadata": metadata, "payload": copy.deepcopy(payload)}


def build_m3_request(
    case: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT_PATH,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT_PATH,
) -> dict[str, Any]:
    """Build one zero-shot request object; never sends it."""

    target = _validate_case(case)
    contract = load_contract(config_path, m3_prompt_path, m4_prompt_path)
    messages = [
        {"role": "developer", "content": contract["developer_instruction"]},
        _summary_message(target["fact_summary"]),
    ]
    payload = _base_payload(contract, messages)
    return _result(
        method="M3",
        target=target,
        contract=contract,
        payload=payload,
        demo_bank_version=None,
        demo_bank_sha256=None,
    )


def build_m4_request(
    case: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]] | None,
    *,
    demo_bank_version: str | None,
    heldout_jurisdictions: Sequence[str] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT_PATH,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT_PATH,
) -> dict[str, Any]:
    """Build one six-shot request object, failing closed on any demo defect."""

    target = _validate_case(case)
    normalized_demos, bank_hash = _validate_demo_bank(
        demos, demo_bank_version, heldout_jurisdictions
    )
    if target["search_rank"] in {item["search_rank"] for item in normalized_demos}:
        raise RequestBuildError("Target case is one of the six demonstrations")
    if target["canonical_url"] in {item["canonical_url"] for item in normalized_demos}:
        raise RequestBuildError("Target canonical URL is one of the six demonstrations")

    contract = load_contract(config_path, m3_prompt_path, m4_prompt_path)
    messages: list[dict[str, str]] = [
        {"role": "developer", "content": contract["developer_instruction"]}
    ]
    for demo in normalized_demos:
        messages.append(_summary_message(demo["fact_summary"]))
        messages.append(_assistant_demo_message(demo["output"]))
    messages.append(_summary_message(target["fact_summary"]))
    payload = _base_payload(contract, messages)
    return _result(
        method="M4",
        target=target,
        contract=contract,
        payload=payload,
        demo_bank_version=demo_bank_version,
        demo_bank_sha256=bank_hash,
    )


def build_request(
    method: str,
    case: Mapping[str, Any],
    *,
    demos: Sequence[Mapping[str, Any]] | None = None,
    demo_bank_version: str | None = None,
    heldout_jurisdictions: Sequence[str] | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT_PATH,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT_PATH,
) -> dict[str, Any]:
    """Dispatch to the frozen M3 or M4 preparation-only builder."""

    normalized_method = method.strip().upper() if isinstance(method, str) else ""
    if normalized_method == "M3":
        if demos is not None or demo_bank_version is not None:
            raise RequestBuildError("M3 must not receive demonstrations or a demo bank")
        return build_m3_request(
            case,
            config_path=config_path,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
        )
    if normalized_method == "M4":
        return build_m4_request(
            case,
            demos,
            demo_bank_version=demo_bank_version,
            heldout_jurisdictions=heldout_jurisdictions,
            config_path=config_path,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
        )
    raise RequestBuildError("method must be exactly M3 or M4")


__all__ = [
    "ACT_IDS",
    "MEANS_IDS",
    "PURPOSE_IDS",
    "RequestBuildError",
    "build_m3_request",
    "build_m4_request",
    "build_request",
    "canonical_json",
    "extract_marked_instruction_block",
    "load_contract",
    "load_shared_prompt_contract",
    "validate_structured_output",
]
