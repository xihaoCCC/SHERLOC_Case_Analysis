#!/usr/bin/env python3
"""Shared leakage-free preparation for dedicated Evaluation B M1/M2 runs.

Only label-free Evaluation B membership and the original public Fact Summary are
made available to the model runners. Human AMP labels are never returned by this
module. The supervised fitting source is the frozen 1,263-case SHERLOC
silver-reference AMP benchmark after exclusion of every retained Evaluation B
case, including narrative-insufficiency/ABSTAIN cases.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import socket
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .metrics import AMP_FAMILY_BY_LABEL, AMP_LABEL_IDS
except ImportError:  # pragma: no cover - direct script imports.
    from metrics import AMP_FAMILY_BY_LABEL, AMP_LABEL_IDS  # type: ignore


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_RELIABILITY_SAMPLE = REPO_ROOT / "data/annotations/reliability_sample_100.csv"
DEFAULT_HUMAN_REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
DEFAULT_MEMBERSHIP = (
    REPO_ROOT / "outputs/analysis/evaluation_b/human_grounded_reference_membership_v1.csv"
)
DEFAULT_MEMBERSHIP_FREEZE = (
    REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_membership_manifest.json"
)
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_AUDIT = REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv"
DEFAULT_PREFLIGHT = REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_supervised_preflight.json"

EXPECTED_BENCHMARK_SHA256 = "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
EXPECTED_ONTOLOGY_SHA256 = "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
EXPECTED_RELIABILITY_SAMPLE_SHA256 = "ff825d1996ff55a72030ef07835c3b71c318df4f43a40ecb79714c27192794bc"
EXPECTED_A1_SPLIT_SHA256 = "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"
EXPECTED_A2_SPLIT_SHA256 = "75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67"
EXPECTED_PRIMARY_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9"
)
EXPECTED_PRIMARY_N = 1263
EXPECTED_RELIABILITY_N = 100
RETAINED_STATUSES = ("SUBSTANTIVE", "ABSTAIN")


class EvaluationBSupervisedError(RuntimeError):
    """Raised when the dedicated supervised Evaluation B protocol is unsafe."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    if not path.is_dir():
        raise EvaluationBSupervisedError(f"Directory does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise EvaluationBSupervisedError(f"Directory is empty: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationBSupervisedError(f"Value is not strict finite JSON: {exc}") from exc


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise EvaluationBSupervisedError(f"Required artifact is missing: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    _require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError(f"Malformed JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationBSupervisedError(f"JSON artifact must be an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_file(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EvaluationBSupervisedError(
                        f"Expected JSON object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError(f"Malformed JSONL artifact: {path}") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    _require_file(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvaluationBSupervisedError(f"CSV has no header: {path}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvaluationBSupervisedError(f"Malformed CSV artifact: {path}") from exc


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationBSupervisedError(f"Refusing to write empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationBSupervisedError(f"Refusing to write empty JSONL: {path}")
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def membership_digest(
    rows: Sequence[tuple[str, int, str, str, str]],
) -> str:
    """Match the canonical retained-membership identity frozen for Eval B.

    The same five identity/input fields and canonical JSON serialization are
    used by the reference freezer and M3/M4 runner.  This makes membership
    provenance directly comparable across all four methods.
    """

    payload = [
        {
            "reliability_case_id": case_id,
            "search_rank": rank,
            "canonical_url": canonical_url,
            "input_sha256": input_sha256,
            "review_status": status,
        }
        for case_id, rank, canonical_url, input_sha256, status in sorted(
            rows, key=lambda item: item[1]
        )
    ]
    return sha256_text(canonical_json(payload))


def _truthy(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def _integer(value: Any, *, field: str, source: str) -> int:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise EvaluationBSupervisedError(f"{source}.{field} is not an integer: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise EvaluationBSupervisedError(f"{source}.{field} is not an integer: {value!r}")
    return int(numeric)


def _first(row: Mapping[str, Any], names: Sequence[str], *, required: bool = True) -> str:
    for name in names:
        if name in row:
            value = str(row.get(name) or "").strip()
            if value or not required:
                return value
    if required:
        raise EvaluationBSupervisedError(f"Missing required field aliases {list(names)}")
    return ""


def _find_artifact_hash(value: Any, target: Path) -> str | None:
    """Find a path/hash pair in a freeze manifest without assuming key ordering."""

    target_values = {
        str(target),
        str(target.resolve()),
        target.as_posix(),
        target.name,
    }
    if isinstance(value, Mapping):
        path_values = [
            str(item)
            for key, item in value.items()
            if "path" in str(key).casefold() and isinstance(item, (str, Path))
        ]
        hash_values = [
            str(item)
            for key, item in value.items()
            if "sha256" in str(key).casefold() and isinstance(item, str)
        ]
        if any(path in target_values or Path(path).name == target.name for path in path_values):
            valid = [item for item in hash_values if len(item) == 64]
            if len(valid) == 1:
                return valid[0]
        # Common explicit top-level spellings.
        stem_tokens = (
            ("reference", "human_reference"),
            ("membership", "membership"),
        )
        for target_token, key_token in stem_tokens:
            if target_token in target.name.casefold():
                for key, item in value.items():
                    lowered = str(key).casefold()
                    if key_token in lowered and "sha256" in lowered and isinstance(item, str) and len(item) == 64:
                        return item
        for nested in value.values():
            found = _find_artifact_hash(nested, target)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_artifact_hash(nested, target)
            if found:
                return found
    return None


def _find_count(value: Any, keys: set[str]) -> int | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).casefold() in keys:
                try:
                    return int(item)
                except (TypeError, ValueError):
                    pass
        for nested in value.values():
            found = _find_count(nested, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_count(nested, keys)
            if found is not None:
                return found
    return None


def _freeze_artifact_sha256(
    freeze: Mapping[str, Any], *, key: str, expected_path: Path
) -> str:
    artifact = freeze.get(key)
    if not isinstance(artifact, Mapping):
        raise EvaluationBSupervisedError(
            f"Evaluation B freeze manifest lacks artifact object {key!r}"
        )
    recorded_path = Path(str(artifact.get("path") or ""))
    resolved_path = recorded_path if recorded_path.is_absolute() else REPO_ROOT / recorded_path
    if resolved_path.resolve() != expected_path.resolve():
        raise EvaluationBSupervisedError(
            f"Evaluation B freeze {key} path differs from requested artifact"
        )
    recorded_sha = str(artifact.get("sha256") or "")
    if len(recorded_sha) != 64:
        raise EvaluationBSupervisedError(
            f"Evaluation B freeze {key} SHA-256 is missing or malformed"
        )
    return recorded_sha


@dataclass(frozen=True)
class TargetCase:
    reliability_case_id: str
    search_rank: int
    canonical_url: str
    jurisdiction: str
    fact_summary: str
    review_status: str

    def model_record(self) -> dict[str, Any]:
        """Return a model-compatible record with no human annotation values."""

        return {
            "identity": {
                "search_rank": self.search_rank,
                "canonical_url": self.canonical_url,
                "jurisdiction_country_raw": self.jurisdiction,
            },
            "text_input": {"english_fact_summary_raw": self.fact_summary},
            "amp_targets": {
                "act_ontology_ids": [],
                "means_ontology_ids": [],
                "purpose_ontology_ids": [],
            },
        }


@dataclass(frozen=True)
class PreparedSupervisedData:
    benchmark: tuple[dict[str, Any], ...]
    training_records: tuple[dict[str, Any], ...]
    target_cases: tuple[TargetCase, ...]
    label_order: tuple[str, ...]
    exclusion_audit: tuple[dict[str, Any], ...]
    source_hashes: Mapping[str, str]
    membership_sha256: str
    retained_primary_cohort_n: int
    training_label_supports: Mapping[str, int]

    @property
    def retained_n(self) -> int:
        return len(self.target_cases)

    @property
    def train_n(self) -> int:
        return len(self.training_records)


def ontology_label_order(ontology: Mapping[str, Any]) -> tuple[str, ...]:
    order = tuple(
        item["id"]
        for family in ("ACT", "MEANS", "PURPOSE")
        for item in ontology["families"][family]
    )
    if order != tuple(AMP_LABEL_IDS):
        raise EvaluationBSupervisedError("Ontology order differs from frozen 17-label AMP order")
    return order


def record_labels(record: Mapping[str, Any]) -> tuple[str, ...]:
    targets = record["amp_targets"]
    labels = tuple(
        targets["act_ontology_ids"]
        + targets["means_ontology_ids"]
        + targets["purpose_ontology_ids"]
    )
    if len(labels) != len(set(labels)) or set(labels) - set(AMP_LABEL_IDS):
        raise EvaluationBSupervisedError("Benchmark row contains duplicate/unknown AMP labels")
    return labels


def target_matrix(records: Sequence[Mapping[str, Any]], label_order: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[int(label in set(record_labels(record))) for label in label_order] for record in records],
        dtype=np.float32,
    )


def texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [str(record["text_input"]["english_fact_summary_raw"]) for record in records]
    if any(not value.strip() for value in result):
        raise EvaluationBSupervisedError("Encountered an empty English Fact Summary")
    return result


def _validate_frozen_source(path: Path, expected_sha256: str, label: str) -> str:
    actual = sha256_file(_require_file(path))
    if actual != expected_sha256:
        raise EvaluationBSupervisedError(
            f"Frozen {label} hash mismatch: expected {expected_sha256}, got {actual}"
        )
    return actual


def prepare_supervised_data(
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    ontology_path: Path = DEFAULT_ONTOLOGY,
    reliability_sample_path: Path = DEFAULT_RELIABILITY_SAMPLE,
    human_reference_path: Path = DEFAULT_HUMAN_REFERENCE,
    membership_path: Path = DEFAULT_MEMBERSHIP,
    membership_freeze_path: Path = DEFAULT_MEMBERSHIP_FREEZE,
    a1_split_path: Path = DEFAULT_A1_SPLIT,
    a2_split_path: Path = DEFAULT_A2_SPLIT,
) -> PreparedSupervisedData:
    """Validate frozen membership and return label-isolated model inputs."""

    source_hashes = {
        "benchmark": _validate_frozen_source(benchmark_path, EXPECTED_BENCHMARK_SHA256, "benchmark"),
        "ontology": _validate_frozen_source(ontology_path, EXPECTED_ONTOLOGY_SHA256, "ontology"),
        "reliability_sample": _validate_frozen_source(
            reliability_sample_path, EXPECTED_RELIABILITY_SAMPLE_SHA256, "reliability sample"
        ),
        "a1_split": _validate_frozen_source(a1_split_path, EXPECTED_A1_SPLIT_SHA256, "A1 split"),
        "a2_split": _validate_frozen_source(a2_split_path, EXPECTED_A2_SPLIT_SHA256, "A2 split"),
    }
    reference_sha = sha256_file(_require_file(human_reference_path))
    membership_sha = sha256_file(_require_file(membership_path))
    freeze_sha = sha256_file(_require_file(membership_freeze_path))
    source_hashes |= {
        "human_reference": reference_sha,
        "membership": membership_sha,
        "membership_freeze": freeze_sha,
    }

    freeze = load_json(membership_freeze_path)
    if freeze.get("status") != "FROZEN_FOR_EVALUATION_B_PRE_MODEL_INFERENCE":
        raise EvaluationBSupervisedError(
            "Evaluation B membership manifest is not explicitly FROZEN"
        )
    expected_reference_hash = _freeze_artifact_sha256(
        freeze, key="human_reference", expected_path=human_reference_path
    )
    expected_membership_hash = _freeze_artifact_sha256(
        freeze, key="membership", expected_path=membership_path
    )
    if expected_reference_hash != reference_sha:
        raise EvaluationBSupervisedError(
            "Human-reference bytes do not match the membership freeze manifest"
        )
    if expected_membership_hash != membership_sha:
        raise EvaluationBSupervisedError(
            "Label-free membership bytes do not match the membership freeze manifest"
        )

    membership_rows = load_csv(membership_path)
    if not membership_rows:
        raise EvaluationBSupervisedError("Evaluation B membership is empty")
    retained_membership: list[dict[str, str]] = []
    for row in membership_rows:
        retained_value = row.get("retained")
        retained = _truthy(retained_value) if retained_value is not None else (
            str(row.get("review_status") or "").strip().upper() in RETAINED_STATUSES
        )
        if retained:
            retained_membership.append(row)
    if not retained_membership:
        raise EvaluationBSupervisedError("Membership contains no retained cases")

    membership_by_id: dict[str, dict[str, str]] = {}
    membership_by_rank: dict[int, dict[str, str]] = {}
    for row in retained_membership:
        case_id = _first(row, ("reliability_case_id",))
        rank = _integer(_first(row, ("search_rank",)), field="search_rank", source=case_id)
        status = _first(row, ("review_status",)).upper()
        if status not in RETAINED_STATUSES:
            raise EvaluationBSupervisedError(f"Retained case {case_id} has status {status!r}")
        if case_id in membership_by_id or rank in membership_by_rank:
            raise EvaluationBSupervisedError(f"Duplicate retained case identity: {case_id}/{rank}")
        membership_by_id[case_id] = row
        membership_by_rank[rank] = row
    if not any(str(row["review_status"]).upper() == "SUBSTANTIVE" for row in retained_membership):
        raise EvaluationBSupervisedError("Retained membership contains no SUBSTANTIVE case")
    if not any(str(row["review_status"]).upper() == "ABSTAIN" for row in retained_membership):
        raise EvaluationBSupervisedError("Retained membership contains no ABSTAIN case")

    frozen_retained_n = _find_count(
        freeze, {"retained_n", "retained_total_n", "retained_total", "retained_count"}
    )
    if frozen_retained_n is not None and frozen_retained_n != len(retained_membership):
        raise EvaluationBSupervisedError(
            f"Freeze retained N={frozen_retained_n}, membership N={len(retained_membership)}"
        )

    # The reference is read only for identity/status equivalence. No human-label
    # field is selected or returned to either model runner.
    reference_rows = load_csv(human_reference_path)
    reference_identity: dict[str, tuple[int, str, str, str]] = {}
    for row in reference_rows:
        case_id = _first(row, ("reliability_case_id",))
        rank = _integer(_first(row, ("search_rank",)), field="search_rank", source=case_id)
        status = _first(row, ("review_status",)).upper()
        canonical_url = _first(row, ("canonical_url",))
        input_sha256 = _first(row, ("input_sha256",))
        if case_id in reference_identity:
            raise EvaluationBSupervisedError(f"Duplicate human-reference case ID: {case_id}")
        reference_identity[case_id] = (rank, status, canonical_url, input_sha256)
    expected_reference_identity = {
        case_id: (
            _integer(_first(row, ("search_rank",)), field="search_rank", source=case_id),
            _first(row, ("review_status",)).upper(),
            _first(row, ("canonical_url",)),
            _first(row, ("input_sha256",)),
        )
        for case_id, row in membership_by_id.items()
    }
    if reference_identity != expected_reference_identity:
        raise EvaluationBSupervisedError(
            "Human-reference and frozen retained membership identities/inputs/statuses differ"
        )

    reliability_rows = load_csv(reliability_sample_path)
    if len(reliability_rows) != EXPECTED_RELIABILITY_N:
        raise EvaluationBSupervisedError(
            f"Reliability sample N={len(reliability_rows)}; expected {EXPECTED_RELIABILITY_N}"
        )
    reliability_by_id: dict[str, dict[str, str]] = {}
    for row in reliability_rows:
        case_id = _first(row, ("reliability_case_id",))
        if case_id in reliability_by_id:
            raise EvaluationBSupervisedError(f"Duplicate reliability case ID: {case_id}")
        reliability_by_id[case_id] = row
    if not set(membership_by_id).issubset(reliability_by_id):
        raise EvaluationBSupervisedError(
            f"Retained cases missing from reliability sample: {sorted(set(membership_by_id) - set(reliability_by_id))}"
        )

    target_cases: list[TargetCase] = []
    membership_tuples: list[tuple[str, int, str, str, str]] = []
    for case_id, member in membership_by_id.items():
        source = reliability_by_id[case_id]
        rank = _integer(source["search_rank"], field="search_rank", source=case_id)
        member_rank = _integer(member["search_rank"], field="search_rank", source=case_id)
        if rank != member_rank:
            raise EvaluationBSupervisedError(f"Rank mismatch for retained case {case_id}")
        status = _first(member, ("review_status",)).upper()
        canonical_url = _first(source, ("canonical_url",))
        member_url = _first(member, ("canonical_url",), required=False)
        if member_url and member_url != canonical_url:
            raise EvaluationBSupervisedError(f"Canonical URL mismatch for retained case {case_id}")
        fact_summary = _first(source, ("english_fact_summary_raw", "fact_summary"))
        jurisdiction = _first(source, ("jurisdiction_raw", "jurisdiction"))
        input_sha256 = sha256_text(fact_summary)
        frozen_input_sha256 = _first(member, ("input_sha256",))
        if frozen_input_sha256 != input_sha256:
            raise EvaluationBSupervisedError(
                f"Frozen narrative hash mismatch for retained case {case_id}"
            )
        target_cases.append(
            TargetCase(case_id, rank, canonical_url, jurisdiction, fact_summary, status)
        )
        membership_tuples.append(
            (case_id, rank, canonical_url, input_sha256, status)
        )
    target_cases.sort(key=lambda case: case.search_rank)
    retained_membership_sha = membership_digest(membership_tuples)
    frozen_digest = str(freeze.get("retained_membership_sha256") or "")
    if frozen_digest != retained_membership_sha:
        raise EvaluationBSupervisedError(
            "Retained membership identity digest differs from freeze manifest"
        )

    benchmark = load_jsonl(benchmark_path)
    if len(benchmark) != EXPECTED_PRIMARY_N:
        raise EvaluationBSupervisedError(
            f"Benchmark N={len(benchmark)}; expected {EXPECTED_PRIMARY_N}"
        )
    benchmark_by_rank: dict[int, dict[str, Any]] = {}
    for record in benchmark:
        if record.get("primary_cohort_id") != EXPECTED_PRIMARY_COHORT_ID:
            raise EvaluationBSupervisedError("Benchmark cohort ID differs from frozen cohort")
        rank = int(record["identity"]["search_rank"])
        if rank in benchmark_by_rank:
            raise EvaluationBSupervisedError(f"Duplicate benchmark rank: {rank}")
        benchmark_by_rank[rank] = record

    retained_ranks = {case.search_rank for case in target_cases}
    training_records = tuple(
        record for rank, record in sorted(benchmark_by_rank.items()) if rank not in retained_ranks
    )
    retained_primary_n = len(retained_ranks & set(benchmark_by_rank))
    if len(training_records) != EXPECTED_PRIMARY_N - retained_primary_n:
        raise EvaluationBSupervisedError("Leakage-free training count arithmetic failed")
    if retained_ranks & {int(record["identity"]["search_rank"]) for record in training_records}:
        raise EvaluationBSupervisedError("Retained Evaluation B case leaked into training records")

    ontology = load_json(ontology_path)
    label_order = ontology_label_order(ontology)
    y_train = target_matrix(training_records, label_order)
    supports = {label: int(y_train[:, index].sum()) for index, label in enumerate(label_order)}
    missing_support = [label for label, support in supports.items() if support == 0]
    if missing_support:
        raise EvaluationBSupervisedError(
            f"Leakage-free training data has zero support for labels: {missing_support}"
        )

    a1_rows = load_csv(a1_split_path)
    a1_by_rank = {int(row["search_rank"]): row for row in a1_rows}
    a2_rows = load_csv(a2_split_path)
    a2_by_rank_fold = {
        (int(row["search_rank"]), int(row["fold_id"])): row for row in a2_rows
    }
    audit: list[dict[str, Any]] = []
    for case in sorted(target_cases, key=lambda value: value.reliability_case_id):
        a1 = a1_by_rank.get(case.search_rank, {})
        a2 = {fold: a2_by_rank_fold.get((case.search_rank, fold), {}) for fold in (1, 2, 3)}
        demo_roles = [
            str(a1.get("demo_bank_role") or ""),
            *(str(a2[fold].get("demo_bank_role") or "") for fold in (1, 2, 3)),
            *(str(a2[fold].get("approved_demo_pool_role") or "") for fold in (1, 2, 3)),
        ]
        demo_status = next((role for role in demo_roles if role), "NONE")
        audit.append(
            {
                "reliability_case_id": case.reliability_case_id,
                "search_rank": case.search_rank,
                "canonical_url": case.canonical_url,
                "jurisdiction": case.jurisdiction,
                "review_status": case.review_status,
                "retained_evaluation_b_case": "TRUE",
                "primary_silver_amp_cohort_member": "TRUE" if case.search_rank in benchmark_by_rank else "FALSE",
                "original_a1_split_membership": a1.get("split", "OUTSIDE_PRIMARY_AMP_COHORT"),
                "original_a1_effective_supervised_train": a1.get("effective_supervised_train", "0"),
                "original_a1_demo_bank_role": a1.get("demo_bank_role", ""),
                "a2_fold_1_role": a2[1].get("role", "OUTSIDE_PRIMARY_AMP_COHORT"),
                "a2_fold_2_role": a2[2].get("role", "OUTSIDE_PRIMARY_AMP_COHORT"),
                "a2_fold_3_role": a2[3].get("role", "OUTSIDE_PRIMARY_AMP_COHORT"),
                "active_or_reserve_demo_status": demo_status,
                "removed_from_eval_b_supervised_training": "TRUE",
                "removed_from_eval_b_validation": "TRUE",
                "removed_from_eval_b_threshold_tuning": "TRUE",
                "removed_from_eval_b_supervised_label_selection": "TRUE",
                "removal_reason": "RETAINED_EVALUATION_B_CASE",
                "membership_sha256": retained_membership_sha,
                "membership_freeze_sha256": freeze_sha,
                "benchmark_sha256": source_hashes["benchmark"],
            }
        )

    return PreparedSupervisedData(
        benchmark=tuple(benchmark),
        training_records=training_records,
        target_cases=tuple(target_cases),
        label_order=label_order,
        exclusion_audit=tuple(audit),
        source_hashes=source_hashes,
        membership_sha256=retained_membership_sha,
        retained_primary_cohort_n=retained_primary_n,
        training_label_supports=supports,
    )


def write_preflight_artifacts(
    prepared: PreparedSupervisedData,
    *,
    audit_path: Path = DEFAULT_AUDIT,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    config_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    atomic_csv(audit_path, prepared.exclusion_audit)
    substantive_n = sum(case.review_status == "SUBSTANTIVE" for case in prepared.target_cases)
    abstain_n = sum(case.review_status == "ABSTAIN" for case in prepared.target_cases)
    summary = {
        "preflight_schema_version": "sherloc-eval-b-supervised-preflight-v1",
        "generator_version": VERSION,
        "created_at": utc_now(),
        "status": "READY_FOR_FIXED_SUPERVISED_EXECUTION_PENDING_EXPLICIT_CONFIRMATION",
        "retained_n": prepared.retained_n,
        "substantive_n": substantive_n,
        "abstain_n": abstain_n,
        "primary_silver_cohort_n": EXPECTED_PRIMARY_N,
        "retained_primary_silver_cohort_n": prepared.retained_primary_cohort_n,
        "dedicated_supervised_train_n": prepared.train_n,
        "retained_in_training_n": 0,
        "all_retained_removed_from_training_validation_tuning_and_selection": True,
        "training_target": "SHERLOC_LEGACY_KEYWORDS_SILVER_REFERENCE_AMP",
        "prediction_input": "ENGLISH_FACT_SUMMARY_ONLY",
        "human_labels_available_to_runner": False,
        "human_labels_used_for_training_tuning_or_selection": False,
        "hyperparameter_search": False,
        "threshold_search": False,
        "training_label_supports": dict(prepared.training_label_supports),
        "retained_membership_sha256": prepared.membership_sha256,
        "source_sha256": dict(prepared.source_hashes),
        "config_sha256": dict(config_hashes or {}),
        "exclusion_audit_path": str(audit_path.resolve()),
        "exclusion_audit_sha256": sha256_file(audit_path),
        "execution_gate": "REQUIRES_ROOT_CONFIRMATION_OF_QC_AND_MEMBERSHIP_FREEZE",
    }
    atomic_json(preflight_path, summary)
    return summary


def prediction_rows(
    *,
    method_id: str,
    target_cases: Sequence[TargetCase],
    probabilities: np.ndarray,
    label_order: Sequence[str],
    threshold: float,
    run_id: str,
    config_sha256: str,
    membership_sha256: str,
    training_membership_sha256: str,
    truncated_by_rank: Mapping[int, bool] | None = None,
    original_token_count_by_rank: Mapping[int, int] | None = None,
) -> list[dict[str, Any]]:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.shape != (len(target_cases), len(label_order)):
        raise EvaluationBSupervisedError(
            f"Prediction matrix shape {matrix.shape}; expected {(len(target_cases), len(label_order))}"
        )
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise EvaluationBSupervisedError("Prediction probabilities are not finite [0,1]")
    rows: list[dict[str, Any]] = []
    for case, scores in zip(target_cases, matrix, strict=True):
        predicted = [label for label, score in zip(label_order, scores, strict=True) if score >= threshold]
        rows.append(
            {
                "prediction_schema_version": "sherloc-eval-b-amp-prediction-v1",
                "run_id": run_id,
                "method_id": method_id,
                "evaluation": "B",
                "cohort": "EVAL_B_RETAINED_SINGLE_REVIEWER",
                "reliability_case_id": case.reliability_case_id,
                "search_rank": case.search_rank,
                "canonical_url": case.canonical_url,
                "jurisdiction": case.jurisdiction,
                "fact_summary": case.fact_summary,
                "input_sha256": sha256_text(case.fact_summary),
                "predicted_labels": predicted,
                "probabilities_by_label": {
                    label: float(score) for label, score in zip(label_order, scores, strict=True)
                },
                "selected_threshold": threshold,
                "truncated_input": bool((truncated_by_rank or {}).get(case.search_rank, False)),
                "original_token_count": (original_token_count_by_rank or {}).get(case.search_rank),
                "config_sha256": config_sha256,
                "retained_membership_sha256": membership_sha256,
                "training_membership_sha256": training_membership_sha256,
                "human_labels_used_for_training_tuning_or_prediction": False,
                "status": "SUCCESS_VALIDATED",
            }
        )
    return sorted(rows, key=lambda row: int(row["search_rank"]))


@dataclass
class RunLock:
    path: Path
    token: str

    @classmethod
    def acquire(cls, model_dir: Path) -> "RunLock":
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "run.lock.json"
        token = os.urandom(16).hex()
        payload = {
            "schema_version": "sherloc-eval-b-supervised-lock-v1",
            "token": token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise EvaluationBSupervisedError(
                f"Evaluation B supervised run lock already exists: {path}"
            ) from exc
        try:
            os.write(descriptor, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return cls(path, token)

    def release(self) -> None:
        payload = load_json(self.path)
        if payload.get("token") != self.token:
            raise EvaluationBSupervisedError(f"Run lock ownership changed: {self.path}")
        self.path.unlink()


def training_membership_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        "".join(
            f"{int(record['identity']['search_rank'])}\t{record['identity']['canonical_url']}\n"
            for record in sorted(records, key=lambda row: int(row["identity"]["search_rank"]))
        )
    )
