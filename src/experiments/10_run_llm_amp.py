#!/usr/bin/env python3
"""Run the frozen Phase-4 M3/M4 AMP extraction experiments.

The runner has deliberately separate, fail-closed stages:

1. M2 A1 and all three M2 A2 runs must be complete and validated before any
   live LLM dry-run request.
2. ``--check-model-access`` freezes the model identifier available to the
   current API project (a dated ``gpt-5.6-luna`` snapshot when exposed,
   otherwise the frozen alias).
3. ``--dry-run`` sends three to five deterministic *non-test* cases and writes
   a technical gate only when every response passes the strict AMP schema.
4. normal execution sends one independent request per frozen A1/A2 TEST case,
   enforcing M3 A1 before M4 A1, complete canonical M1--M4 A1 metrics before
   A2, and all three M3 A2 folds before any M4 A2 request.

Successful responses are first committed as atomic per-case JSON records.
Canonical JSONL files are materialized from those records, making interruption
and resumption safe without resending validated successes.  Failed attempts are
retained separately and are never converted into empty predictions.

Each setting is protected by an exclusive inter-process run lock.  Security
boundary: the module reads credentials only from ``OPENAI_API_KEY`` at
execution time.  The SDK import is lazy, and neither request artifacts nor logs
contain the credential.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import sys
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # Package import.
    from . import llm_request_builder as builder
except ImportError:  # pragma: no cover - direct CLI execution.
    EXPERIMENT_DIR = Path(__file__).resolve().parent
    if str(EXPERIMENT_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENT_DIR))
    import llm_request_builder as builder  # type: ignore


VERSION = "1.2.0"
PREDICTION_SCHEMA_VERSION = "sherloc-amp-predictions-v1"
EXECUTION_SCHEMA_VERSION = "sherloc-llm-amp-execution-v1"
FAILURE_SCHEMA_VERSION = "sherloc-llm-amp-failure-v1"
MODEL_ACCESS_SCHEMA_VERSION = "sherloc-openai-model-access-v1"
DRY_RUN_GATE_SCHEMA_VERSION = "sherloc-llm-dry-run-gate-v1"
RUN_LOCK_SCHEMA_VERSION = "sherloc-llm-setting-run-lock-v1"
FALLBACK_RESERVATION_SCHEMA_VERSION = "sherloc-llm-fallback-reservation-v1"
PRIMARY_RECOVERY_RESERVATION_SCHEMA_VERSION = (
    "sherloc-llm-primary-recovery-reservation-v1"
)
EXPECTED_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_BENCHMARK_N = 1263
EXPECTED_BENCHMARK_SHA256 = (
    "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
)
EXPECTED_A1_SPLIT_SHA256 = (
    "63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d"
)
EXPECTED_A2_SPLIT_SHA256 = (
    "75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67"
)
EXPECTED_CONFIG_SHA256 = (
    "5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b"
)
EXPECTED_DEMO_BANK_SHA256 = (
    "1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b"
)
EXPECTED_M3_PROMPT_SHA256 = (
    "00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85"
)
EXPECTED_M4_PROMPT_SHA256 = (
    "2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
)
EXPECTED_REVIEW_SHA256 = (
    "c7e793e781c77bde4f99507b66b6ffeb5e37de768c86fd27f58c9e5cdf5e242f"
)
EXPECTED_APPROVED_RANKS = (1487, 1494, 1178, 498, 391, 157, 1343, 936)
EXPECTED_ACTIVE_RANKS = EXPECTED_APPROVED_RANKS[:6]
EXPECTED_RESERVE_RANKS = EXPECTED_APPROVED_RANKS[6:]
EXPECTED_REJECTED_RANKS = (146, 1211)
MODEL_ALIAS = "gpt-5.6-luna"
DATED_MODEL_PATTERN = re.compile(r"^gpt-5\.6-luna-\d{4}-\d{2}-\d{2}$")
DRY_RUN_SEED = 20260813
MIN_DRY_RUN_CASES = 3
MAX_DRY_RUN_CASES = 5
DEFAULT_DRY_RUN_CASES = 5
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0
DEFAULT_WORKERS = 1
MAX_WORKERS = 8
MALFORMED_LOCK_STALE_SECONDS = 24 * 60 * 60
TECHNICAL_AMENDMENT_ID = "sherloc-llm-amp-technical-failure-amendment-v1"
INITIAL_MAX_OUTPUT_TOKENS = 512
FALLBACK_MAX_OUTPUT_TOKENS = 2048
MAX_FALLBACK_ATTEMPTS_PER_CASE = 2
EXPECTED_M3_A1_AMENDMENT_PENDING_RANKS = frozenset({266, 551, 1356})
RANK_1340_EXCEPTION_ID = (
    "sherloc-m4-a2-fold1-rank1340-rate-limit-exception-v1"
)
RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS = 4

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/llm_extraction_amp_v2.yaml"
DEFAULT_DEMO_BANK = REPO_ROOT / "config/experiments/demo_bank_amp_v1.yaml"
DEFAULT_M3_PROMPT = REPO_ROOT / "prompts/m3_zero_shot_amp_v2.md"
DEFAULT_M4_PROMPT = REPO_ROOT / "prompts/m4_six_shot_amp_v2.md"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_REVIEW = REPO_ROOT / "data/annotations/demo_bank_review_v2.csv"
DEFAULT_TECHNICAL_AMENDMENT = (
    REPO_ROOT / "docs/llm_amp_technical_failure_amendment_v1.md"
)
DEFAULT_RANK_1340_EXCEPTION_ADDENDUM = (
    REPO_ROOT / "docs/m4_a2_rank_1340_technical_exception_addendum_v1.md"
)
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions"
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs/logs/llm"
DEFAULT_METRIC_ROOT = REPO_ROOT / "outputs/metrics"
DEFAULT_M2_MODEL_ROOT = REPO_ROOT / "outputs/models/m2"
DEFAULT_M2_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions/m2"
DEFAULT_M2_CONFIG = REPO_ROOT / "config/experiments/m2_modernbert_amp_v2.yaml"
EXPECTED_M2_CONFIG_SHA256 = (
    "73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3"
)
EXPECTED_TECHNICAL_AMENDMENT_SHA256 = (
    "363c06abb49390a3cf66d646466313d6f50d655e41b801483063d1b180d7cb84"
)
EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256 = (
    "0ebb7945049d097476c3244407bff46b9f272704eb1a10118e649bfed2c8f6dc"
)

AMP_LABEL_IDS = builder.ACT_IDS + builder.MEANS_IDS + builder.PURPOSE_IDS


class LLMProtocolError(RuntimeError):
    """Raised when execution would violate a frozen or security invariant."""


class MaxOutputTokensIncomplete(LLMProtocolError):
    """Exact technical fallback trigger returned by the Responses API."""

    def __init__(self, response: Any) -> None:
        super().__init__(
            "Response status is 'incomplete'; "
            "details={'reason': 'max_output_tokens'}"
        )
        self.response = response


@dataclass(frozen=True)
class RunSpec:
    method: str
    evaluation: str
    fold: int | None
    dry_run: bool
    bank_id: str | None
    output_path: Path
    state_dir: Path
    diagnostics_path: Path
    failure_manifest_path: Path

    @property
    def setting_id(self) -> str:
        if self.dry_run:
            return f"dry_run_{self.method.lower()}"
        if self.evaluation == "A1":
            return f"{self.method.lower()}_a1"
        return f"{self.method.lower()}_a2_fold_{self.fold}"

    @property
    def split_or_fold(self) -> str:
        if self.dry_run:
            return "A1_NON_TEST_TECHNICAL_DRY_RUN"
        if self.evaluation == "A1":
            return "A1_TEST"
        return f"A2_FOLD_{self.fold}_TEST"


def _is_rank_1340_exception_scope(
    spec: RunSpec, case: Mapping[str, Any]
) -> bool:
    return (
        not spec.dry_run
        and spec.method == "M4"
        and spec.evaluation == "A2"
        and spec.fold == 1
        and int(case.get("search_rank") or 0) == 1340
    )


def _validate_exception_record_scope(
    spec: RunSpec, case: Mapping[str, Any], technical: Mapping[str, Any]
) -> None:
    if technical.get("technical_exception_id") is None:
        return
    if not (
        _is_rank_1340_exception_scope(spec, case)
        and technical.get("technical_exception_id") == RANK_1340_EXCEPTION_ID
        and technical.get("technical_exception_sha256")
        == EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
    ):
        raise LLMProtocolError("Technical exception escaped M4 A2 Fold 1 rank 1340")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


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
        raise LLMProtocolError(f"Value is not canonical-JSON serializable: {exc}") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(path: Path, expected: str, artifact_name: str) -> None:
    if not path.is_file():
        raise LLMProtocolError(f"Canonical {artifact_name} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise LLMProtocolError(
            f"Canonical {artifact_name} SHA-256 changed: expected {expected}, "
            f"observed {observed}"
        )


def assert_canonical_input_paths(
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    a1_split_path: Path = DEFAULT_A1_SPLIT,
    a2_split_path: Path = DEFAULT_A2_SPLIT,
    config_path: Path = DEFAULT_CONFIG,
    demo_bank_path: Path = DEFAULT_DEMO_BANK,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT,
    ontology_path: Path = DEFAULT_ONTOLOGY,
    review_path: Path = DEFAULT_REVIEW,
) -> None:
    """Reject alternate live inputs, including coordinated copied substitutes."""

    supplied = {
        "benchmark": benchmark_path,
        "A1 split": a1_split_path,
        "A2 split": a2_split_path,
        "LLM config": config_path,
        "demo bank": demo_bank_path,
        "M3 prompt": m3_prompt_path,
        "M4 prompt": m4_prompt_path,
        "ontology": ontology_path,
        "demo review": review_path,
    }
    canonical = {
        "benchmark": DEFAULT_BENCHMARK,
        "A1 split": DEFAULT_A1_SPLIT,
        "A2 split": DEFAULT_A2_SPLIT,
        "LLM config": DEFAULT_CONFIG,
        "demo bank": DEFAULT_DEMO_BANK,
        "M3 prompt": DEFAULT_M3_PROMPT,
        "M4 prompt": DEFAULT_M4_PROMPT,
        "ontology": DEFAULT_ONTOLOGY,
        "demo review": DEFAULT_REVIEW,
    }
    mismatches = {
        name: {"expected": str(canonical[name]), "observed": str(path)}
        for name, path in supplied.items()
        if path.resolve() != canonical[name].resolve()
    }
    if mismatches:
        raise LLMProtocolError(
            "Canonical API execution rejects alternate input paths: "
            + canonical_json(mismatches)
        )


def validate_canonical_artifact_hashes() -> None:
    """Validate every immutable artifact before any live API operation."""

    artifacts = (
        (DEFAULT_BENCHMARK, EXPECTED_BENCHMARK_SHA256, "benchmark JSONL"),
        (DEFAULT_A1_SPLIT, EXPECTED_A1_SPLIT_SHA256, "final A1 split"),
        (DEFAULT_A2_SPLIT, EXPECTED_A2_SPLIT_SHA256, "final A2 split"),
        (DEFAULT_CONFIG, EXPECTED_CONFIG_SHA256, "LLM config"),
        (DEFAULT_DEMO_BANK, EXPECTED_DEMO_BANK_SHA256, "demo bank"),
        (DEFAULT_M3_PROMPT, EXPECTED_M3_PROMPT_SHA256, "M3 prompt"),
        (DEFAULT_M4_PROMPT, EXPECTED_M4_PROMPT_SHA256, "M4 prompt"),
        (DEFAULT_ONTOLOGY, EXPECTED_ONTOLOGY_SHA256, "AMP ontology"),
        (DEFAULT_REVIEW, EXPECTED_REVIEW_SHA256, "human demo review"),
        (
            DEFAULT_TECHNICAL_AMENDMENT,
            EXPECTED_TECHNICAL_AMENDMENT_SHA256,
            "LLM technical-failure amendment",
        ),
        (
            DEFAULT_RANK_1340_EXCEPTION_ADDENDUM,
            EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256,
            "M4 A2 rank-1340 technical exception addendum",
        ),
    )
    for path, expected, name in artifacts:
        require_file_sha256(path, expected, name)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _assert_secret_absent(value: Any, secret: str | None) -> None:
    """Fail closed if a credential would be serialized."""

    if secret and secret in canonical_json(value):
        raise LLMProtocolError("Refusing to serialize an API credential")


def atomic_json(path: Path, value: Any, *, secret: str | None = None) -> None:
    _assert_secret_absent(value, secret)
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, secret: str | None = None
) -> None:
    _assert_secret_absent(rows, secret)
    _atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMProtocolError(f"Cannot read JSON-compatible document {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMProtocolError(f"Expected an object in {path}")
    return value


def _pid_is_alive(pid: int) -> bool:
    """Return whether a same-host PID exists without sending it a signal."""

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class SettingRunLock:
    """Exclusive, crash-safe lock for one paid execution setting.

    A persistent guard file is locked with ``flock`` before the human-readable
    marker is inspected or created.  The kernel releases the guard on process
    exit, so stale-marker recovery cannot race another legitimate owner.  The
    marker itself is still created with ``O_EXCL`` and archived on conservative
    stale recovery to retain an audit trail.
    """

    def __init__(self, spec: RunSpec) -> None:
        self.spec = spec
        self.guard_path = spec.state_dir / ".run.guard"
        self.marker_path = spec.state_dir / ".run.lock.json"
        self.history_dir = spec.state_dir / ".lock_history"
        self.stale_dir = spec.state_dir / ".stale_locks"
        self.token = uuid.uuid4().hex
        self.hostname = socket.gethostname()
        self.acquired_at = utc_now()
        self.history_path = self.history_dir / f"{self.token}.json"
        self._guard: Any | None = None
        self._history: dict[str, Any] | None = None

    def _archive_stale_marker(self, reason: str) -> dict[str, Any]:
        self.stale_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = self.stale_dir / f"{stamp}_{self.token}.json"
        os.replace(self.marker_path, target)
        return {
            "reason": reason,
            "archived_at": utc_now(),
            "archived_path": str(target),
            "archived_sha256": sha256_file(target),
        }

    def _recover_or_reject_existing_marker(self) -> dict[str, Any] | None:
        if not self.marker_path.exists():
            return None
        try:
            marker = load_json(self.marker_path)
        except LLMProtocolError as exc:
            try:
                age = max(0.0, time.time() - self.marker_path.stat().st_mtime)
            except OSError as stat_exc:
                raise LLMProtocolError(
                    f"Cannot inspect existing run-lock marker {self.marker_path}: "
                    f"{stat_exc}"
                ) from stat_exc
            if age < MALFORMED_LOCK_STALE_SECONDS:
                raise LLMProtocolError(
                    f"Malformed run-lock marker is too recent for safe recovery: "
                    f"{self.marker_path}; age_seconds={age:.1f}"
                ) from exc
            return self._archive_stale_marker(
                f"MALFORMED_MARKER_OLDER_THAN_{MALFORMED_LOCK_STALE_SECONDS}_SECONDS"
            )

        marker_host = marker.get("hostname")
        marker_pid = marker.get("pid")
        marker_token = marker.get("token")
        if (
            not isinstance(marker_host, str)
            or not marker_host
            or isinstance(marker_pid, bool)
            or not isinstance(marker_pid, int)
            or marker_pid <= 0
            or not isinstance(marker_token, str)
            or not marker_token
        ):
            try:
                age = max(0.0, time.time() - self.marker_path.stat().st_mtime)
            except OSError as exc:
                raise LLMProtocolError(
                    f"Cannot inspect malformed run-lock marker {self.marker_path}: {exc}"
                ) from exc
            if age < MALFORMED_LOCK_STALE_SECONDS:
                raise LLMProtocolError(
                    f"Invalid run-lock marker is too recent for safe recovery: "
                    f"{self.marker_path}; age_seconds={age:.1f}"
                )
            return self._archive_stale_marker(
                f"INVALID_MARKER_OLDER_THAN_{MALFORMED_LOCK_STALE_SECONDS}_SECONDS"
            )
        if marker_host != self.hostname:
            raise LLMProtocolError(
                "Existing run-lock marker belongs to another host and cannot be "
                f"safely declared stale: host={marker_host!r}, path={self.marker_path}"
            )
        if _pid_is_alive(marker_pid):
            raise LLMProtocolError(
                f"Existing run-lock marker names a live process: pid={marker_pid}, "
                f"path={self.marker_path}"
            )
        return self._archive_stale_marker("SAME_HOST_OWNER_PID_NOT_ALIVE")

    def acquire(self) -> "SettingRunLock":
        self.spec.state_dir.mkdir(parents=True, exist_ok=True)
        self.guard_path.touch(mode=0o600, exist_ok=True)
        guard = self.guard_path.open("a+b")
        marker_created = False
        try:
            try:
                fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LLMProtocolError(
                    f"Another process is already executing setting "
                    f"{self.spec.setting_id}: {self.guard_path}"
                ) from exc
            self._guard = guard
            stale = self._recover_or_reject_existing_marker()
            marker = {
                "lock_schema_version": RUN_LOCK_SCHEMA_VERSION,
                "token": self.token,
                "setting_id": self.spec.setting_id,
                "method": self.spec.method,
                "evaluation": self.spec.evaluation,
                "fold": self.spec.fold,
                "dry_run": self.spec.dry_run,
                "pid": os.getpid(),
                "hostname": self.hostname,
                "acquired_at": self.acquired_at,
                "runner_path": str(Path(__file__).resolve()),
                "runner_sha256": sha256_file(Path(__file__).resolve()),
                "guard_path": str(self.guard_path),
            }
            encoded = (
                json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8")
            try:
                descriptor = os.open(
                    self.marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
            except FileExistsError as exc:
                raise LLMProtocolError(
                    f"Run-lock marker appeared concurrently: {self.marker_path}"
                ) from exc
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                marker_created = True
            except Exception:
                if self.marker_path.exists():
                    self.marker_path.unlink()
                raise
            self._history = {
                **marker,
                "status": "ACQUIRED",
                "marker_path": str(self.marker_path),
                "stale_marker_recovery": stale,
            }
            atomic_json(self.history_path, self._history)
            return self
        except Exception:
            if marker_created and self.marker_path.is_file():
                try:
                    owned = load_json(self.marker_path).get("token") == self.token
                except LLMProtocolError:
                    owned = False
                if owned:
                    self.marker_path.unlink()
            if self._guard is not None:
                fcntl.flock(self._guard.fileno(), fcntl.LOCK_UN)
                self._guard.close()
                self._guard = None
            else:
                guard.close()
            raise

    def diagnostic_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_LOCK_SCHEMA_VERSION,
            "token": self.token,
            "setting_id": self.spec.setting_id,
            "pid": os.getpid(),
            "hostname": self.hostname,
            "acquired_at": self.acquired_at,
            "history_path": str(self.history_path),
            "guard_path": str(self.guard_path),
            "marker_path": str(self.marker_path),
        }

    def release(self) -> None:
        if self._guard is None:
            return
        ownership_error: LLMProtocolError | None = None
        try:
            if not self.marker_path.is_file():
                ownership_error = LLMProtocolError(
                    f"Owned run-lock marker disappeared: {self.marker_path}"
                )
            else:
                try:
                    marker = load_json(self.marker_path)
                except LLMProtocolError as exc:
                    ownership_error = exc
                else:
                    if marker.get("token") != self.token:
                        ownership_error = LLMProtocolError(
                            f"Run-lock ownership changed before release: {self.marker_path}"
                        )
                    else:
                        self.marker_path.unlink()
            history = dict(self._history or self.diagnostic_metadata())
            history.update(
                {
                    "status": (
                        "RELEASED" if ownership_error is None else "RELEASE_OWNERSHIP_ERROR"
                    ),
                    "released_at": utc_now(),
                    "release_error": str(ownership_error) if ownership_error else None,
                }
            )
            atomic_json(self.history_path, history)
        finally:
            fcntl.flock(self._guard.fileno(), fcntl.LOCK_UN)
            self._guard.close()
            self._guard = None
        if ownership_error is not None:
            raise ownership_error

    def __enter__(self) -> "SettingRunLock":
        return self.acquire()

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        try:
            self.release()
        except LLMProtocolError:
            if exc_type is None:
                raise
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LLMProtocolError(f"Non-object at {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise LLMProtocolError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise LLMProtocolError(f"Cannot read CSV {path}: {exc}") from exc


def safe_primitive(value: Any) -> Any:
    """Convert SDK/Pydantic values to JSON-compatible values without repr()."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): safe_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_primitive(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return safe_primitive(model_dump(exclude_none=True))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return safe_primitive(to_dict())
    # Unknown SDK objects are not serialized wholesale.  Callers should select
    # explicit scalar attributes instead.
    raise LLMProtocolError(f"Unsupported value for safe serialization: {type(value).__name__}")


def require_api_key(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    key = source.get("OPENAI_API_KEY", "")
    if not key.strip():
        raise LLMProtocolError(
            "OPENAI_API_KEY is absent; configure it in the process environment"
        )
    return key


def load_openai_sdk() -> tuple[Any, str]:
    """Import the official SDK only when a live stage is requested."""

    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise LLMProtocolError(
            "The official openai Python package is required for live execution"
        ) from exc
    version = str(getattr(openai, "__version__", "UNKNOWN"))
    try:
        major_minor = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        major_minor = (0, 0)
    if major_minor < (2, 31):
        raise LLMProtocolError(
            f"OpenAI SDK >=2.31 is required by this runner; observed {version}"
        )
    return openai, version


def create_openai_client(openai_module: Any, api_key: str) -> Any:
    # SDK retries are disabled so retry counts and Retry-After handling are
    # explicit in the research provenance.
    return openai_module.OpenAI(api_key=api_key, max_retries=0)


def benchmark_case(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        identity = record["identity"]
        text = record["text_input"]["english_fact_summary_raw"]
        targets = record["amp_targets"]
        rank = int(identity["search_rank"])
        jurisdiction = str(identity["jurisdiction_country_raw"])
        canonical_url = str(identity["canonical_url"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMProtocolError(f"Malformed benchmark row: {exc}") from exc
    if record.get("primary_cohort_id") != EXPECTED_COHORT_ID:
        raise LLMProtocolError(f"Cohort ID mismatch for benchmark rank {rank}")
    if not isinstance(text, str) or not text.strip():
        raise LLMProtocolError(f"Rank {rank} lacks an English Fact Summary")
    labels = list(
        targets["act_ontology_ids"]
        + targets["means_ontology_ids"]
        + targets["purpose_ontology_ids"]
    )
    if len(labels) != len(set(labels)) or set(labels) - set(AMP_LABEL_IDS):
        raise LLMProtocolError(f"Rank {rank} has invalid AMP silver-reference labels")
    selected = set(labels)
    ordered_labels = [label for label in AMP_LABEL_IDS if label in selected]
    unodc_number = str(identity.get("unodc_case_number") or "").strip()
    return {
        "case_id": unodc_number or f"sherloc-rank-{rank}",
        "search_rank": rank,
        "case_title": str(identity.get("case_title_raw") or ""),
        "canonical_url": canonical_url,
        "jurisdiction": jurisdiction,
        "fact_summary": text,
        "silver_reference_labels": ordered_labels,
    }


def load_benchmark_index(path: Path = DEFAULT_BENCHMARK) -> dict[int, dict[str, Any]]:
    require_file_sha256(path, EXPECTED_BENCHMARK_SHA256, "benchmark JSONL")
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_BENCHMARK_N:
        raise LLMProtocolError(
            f"Expected {EXPECTED_BENCHMARK_N} benchmark rows, got {len(rows)}"
        )
    cases = [benchmark_case(row) for row in rows]
    ranks = [case["search_rank"] for case in cases]
    urls = [case["canonical_url"] for case in cases]
    if len(set(ranks)) != len(cases) or len(set(urls)) != len(cases):
        raise LLMProtocolError("Benchmark ranks or canonical URLs are not unique")
    return {case["search_rank"]: case for case in cases}


def _split_role(row: Mapping[str, str]) -> str:
    return str(row.get("split") or row.get("role") or "").strip().upper()


def _split_labels(row: Mapping[str, str]) -> list[str]:
    if not all(label in row for label in AMP_LABEL_IDS):
        raise LLMProtocolError("Final split is missing AMP target columns")
    invalid = [label for label in AMP_LABEL_IDS if row[label] not in ("0", "1")]
    if invalid:
        raise LLMProtocolError(f"Final split has non-binary target columns: {invalid}")
    return [label for label in AMP_LABEL_IDS if row[label] == "1"]


def load_setting_rows(
    evaluation: str,
    fold: int | None,
    benchmark: Mapping[int, dict[str, Any]],
    *,
    a1_path: Path = DEFAULT_A1_SPLIT,
    a2_path: Path = DEFAULT_A2_SPLIT,
) -> list[dict[str, Any]]:
    if evaluation == "A1":
        if fold is not None:
            raise LLMProtocolError("A1 must not specify a fold")
        require_file_sha256(a1_path, EXPECTED_A1_SPLIT_SHA256, "final A1 split")
        raw_rows = load_csv(a1_path)
    elif evaluation == "A2":
        if fold not in (1, 2, 3):
            raise LLMProtocolError("A2 requires fold 1, 2, or 3")
        require_file_sha256(a2_path, EXPECTED_A2_SPLIT_SHA256, "final A2 split")
        raw_rows = [
            row for row in load_csv(a2_path) if int(row.get("fold_id") or 0) == fold
        ]
    else:
        raise LLMProtocolError("evaluation must be A1 or A2")
    if len(raw_rows) != EXPECTED_BENCHMARK_N:
        raise LLMProtocolError(
            f"{evaluation} fold {fold} contains {len(raw_rows)} rows rather than "
            f"{EXPECTED_BENCHMARK_N}"
        )
    observed = [int(row["search_rank"]) for row in raw_rows]
    if len(set(observed)) != EXPECTED_BENCHMARK_N or set(observed) != set(benchmark):
        raise LLMProtocolError("Final split membership differs from the frozen cohort")
    merged: list[dict[str, Any]] = []
    for row in raw_rows:
        rank = int(row["search_rank"])
        case = benchmark[rank]
        if row.get("canonical_url") != case["canonical_url"]:
            raise LLMProtocolError(f"Canonical URL mismatch for rank {rank}")
        if row.get("jurisdiction") != case["jurisdiction"]:
            raise LLMProtocolError(f"Jurisdiction mismatch for rank {rank}")
        if _split_labels(row) != case["silver_reference_labels"]:
            raise LLMProtocolError(f"Silver-reference mismatch for rank {rank}")
        role = _split_role(row)
        if role not in {"TRAIN", "VALIDATION", "TEST", "ACTIVE_DEMO", "RESERVE_DEMO"}:
            raise LLMProtocolError(f"Unsupported split role {role!r} for rank {rank}")
        merged.append({**case, "role": role})
    return sorted(merged, key=lambda case: case["search_rank"])


def deterministic_dry_run_cases(
    setting_rows: Sequence[Mapping[str, Any]],
    *,
    count: int = DEFAULT_DRY_RUN_CASES,
    excluded_ranks: Iterable[int] = (),
    seed: int = DRY_RUN_SEED,
) -> list[dict[str, Any]]:
    if count < MIN_DRY_RUN_CASES or count > MAX_DRY_RUN_CASES:
        raise LLMProtocolError(
            f"Dry-run count must be {MIN_DRY_RUN_CASES}--{MAX_DRY_RUN_CASES}"
        )
    excluded = set(excluded_ranks)
    eligible = [
        dict(case)
        for case in setting_rows
        if case["role"] != "TEST" and int(case["search_rank"]) not in excluded
    ]
    if len(eligible) < count:
        raise LLMProtocolError("Too few non-test, non-demo cases for the dry run")
    ordered = sorted(
        eligible,
        key=lambda case: (
            sha256_text(f"{seed}:{case['search_rank']}"),
            int(case["search_rank"]),
        ),
    )
    return sorted(ordered[:count], key=lambda case: int(case["search_rank"]))


def validate_frozen_contract(
    *,
    config_path: Path = DEFAULT_CONFIG,
    demo_bank_path: Path = DEFAULT_DEMO_BANK,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT,
    ontology_path: Path = DEFAULT_ONTOLOGY,
    review_path: Path = DEFAULT_REVIEW,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require_file_sha256(
        DEFAULT_TECHNICAL_AMENDMENT,
        EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "LLM technical-failure amendment",
    )
    require_file_sha256(
        DEFAULT_RANK_1340_EXCEPTION_ADDENDUM,
        EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256,
        "M4 A2 rank-1340 technical exception addendum",
    )
    require_file_sha256(config_path, EXPECTED_CONFIG_SHA256, "LLM config")
    require_file_sha256(demo_bank_path, EXPECTED_DEMO_BANK_SHA256, "demo bank")
    require_file_sha256(m3_prompt_path, EXPECTED_M3_PROMPT_SHA256, "M3 prompt")
    require_file_sha256(m4_prompt_path, EXPECTED_M4_PROMPT_SHA256, "M4 prompt")
    require_file_sha256(ontology_path, EXPECTED_ONTOLOGY_SHA256, "AMP ontology")
    require_file_sha256(review_path, EXPECTED_REVIEW_SHA256, "human demo review")
    contract = builder.load_contract(
        config_path=config_path,
        m3_prompt_path=m3_prompt_path,
        m4_prompt_path=m4_prompt_path,
    )
    config = contract["config"]
    if config.get("status") != "FINAL_FROZEN_PRE_MODEL_EXECUTION":
        raise LLMProtocolError("LLM config is not frozen for execution")
    if config.get("primary_cohort_id") != EXPECTED_COHORT_ID:
        raise LLMProtocolError("LLM config cohort ID mismatch")
    if config["api_request"].get("credential_source") != "OPENAI_API_KEY_ENVIRONMENT_ONLY":
        raise LLMProtocolError("Frozen credential source changed")
    if config["api_request"].get("api_key_in_config") is not False:
        raise LLMProtocolError("LLM config may not contain an API key")
    if _contains_secret_key(config):
        raise LLMProtocolError("A credential-like field exists in the frozen config")
    if config["ontology"].get("ontology_sha256") != EXPECTED_ONTOLOGY_SHA256:
        raise LLMProtocolError("LLM config ontology hash differs from the canonical ontology")
    if config["methods"]["M3"].get("prompt_sha256") != EXPECTED_M3_PROMPT_SHA256:
        raise LLMProtocolError("LLM config M3 prompt hash is not canonical")
    if config["methods"]["M4"].get("prompt_sha256") != EXPECTED_M4_PROMPT_SHA256:
        raise LLMProtocolError("LLM config M4 prompt hash is not canonical")
    demo_bank = load_json(demo_bank_path)
    m4 = config["methods"]["M4"]
    if m4.get("demo_bank_file_sha256") != EXPECTED_DEMO_BANK_SHA256:
        raise LLMProtocolError("Frozen demo-bank file hash changed")
    if demo_bank.get("status") != "FINAL_FROZEN_PRE_MODEL_EXECUTION":
        raise LLMProtocolError("Demo bank is not final/frozen")
    if demo_bank.get("bank_version") != m4["demo_bank_version"]:
        raise LLMProtocolError("Demo-bank version mismatch")
    if demo_bank.get("hashes", {}).get("bank_membership_sha256") != m4[
        "demo_bank_membership_sha256"
    ]:
        raise LLMProtocolError("Global demo-bank membership hash mismatch")
    source_review = demo_bank.get("source_review", {})
    if source_review.get("path") != "data/annotations/demo_bank_review_v2.csv":
        raise LLMProtocolError("Demo-bank source review path changed")
    if source_review.get("sha256") != EXPECTED_REVIEW_SHA256:
        raise LLMProtocolError("Demo-bank source review hash changed")
    validate_demo_bank_internal_hashes(demo_bank)
    validate_review_decisions(review_path)
    return contract, config, demo_bank


def validate_review_decisions(review_path: Path = DEFAULT_REVIEW) -> None:
    rows = load_csv(review_path)
    by_rank = {int(row["search_rank"]): row for row in rows}
    keep_ranks = tuple(
        int(row["search_rank"])
        for row in rows
        if row.get("reviewer_approve_v2", "").strip() == "Keep"
    )
    if set(keep_ranks) != set(EXPECTED_APPROVED_RANKS) or len(keep_ranks) != 8:
        raise LLMProtocolError(
            "Human review Keep ranks differ from the exact approved eight"
        )
    for rank in EXPECTED_APPROVED_RANKS:
        if rank not in by_rank or by_rank[rank].get("reviewer_approve_v2", "").strip() != "Keep":
            raise LLMProtocolError(f"Approved rank {rank} is not Keep in human review")
    for rank in EXPECTED_REJECTED_RANKS:
        if rank not in by_rank or by_rank[rank].get("reviewer_approve_v2", "").strip() == "Keep":
            raise LLMProtocolError(f"Rejected rank {rank} is unexpectedly approved")


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {"api_key", "openai_api_key", "authorization"}:
                return True
            if _contains_secret_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


def validate_demo_bank_internal_hashes(demo_bank: Mapping[str, Any]) -> None:
    """Recompute every frozen demo-content and bank-membership digest."""

    approved_raw = demo_bank.get("approved_cases")
    banks = demo_bank.get("evaluation_banks")
    hashes = demo_bank.get("hashes")
    if not isinstance(approved_raw, list) or not isinstance(banks, Mapping):
        raise LLMProtocolError("Malformed frozen demo-bank content")
    if not isinstance(hashes, Mapping):
        raise LLMProtocolError("Frozen demo-bank hash block is absent")
    content_by_rank: dict[int, str] = {}
    approved_without_hash: list[dict[str, Any]] = []
    approved_ranks = tuple(int(item.get("search_rank") or 0) for item in approved_raw)
    if approved_ranks != EXPECTED_APPROVED_RANKS:
        raise LLMProtocolError(
            f"Approved demo ranks/order changed: observed {approved_ranks}"
        )
    roles = demo_bank.get("roles", {})
    if tuple(roles.get("active_six", [])) != EXPECTED_ACTIVE_RANKS:
        raise LLMProtocolError("Frozen active-six ranks/order changed")
    if tuple(roles.get("reserve_two", [])) != EXPECTED_RESERVE_RANKS:
        raise LLMProtocolError("Frozen reserve-two ranks/order changed")
    if set(banks) != {"A1", "A2_FOLD_1", "A2_FOLD_2", "A2_FOLD_3"}:
        raise LLMProtocolError("Frozen evaluation-bank IDs changed")
    for raw in approved_raw:
        if not isinstance(raw, Mapping):
            raise LLMProtocolError("Approved demo entry is not an object")
        item = dict(raw)
        observed = item.pop("case_content_sha256", None)
        expected = sha256_text(canonical_json(item))
        rank = int(item.get("search_rank") or 0)
        if rank <= 0 or observed != expected:
            raise LLMProtocolError(f"Frozen content hash mismatch for demo rank {rank}")
        expected_role = "ACTIVE" if rank in EXPECTED_ACTIVE_RANKS else "RESERVE"
        if item.get("role") != expected_role:
            raise LLMProtocolError(f"Frozen role changed for demo rank {rank}")
        if item.get("human_approved") is not True or item.get("frozen") is not True:
            raise LLMProtocolError(f"Demo rank {rank} lacks approved/frozen flags")
        if item.get("human_approval", {}).get("status") != "Keep":
            raise LLMProtocolError(f"Demo rank {rank} lacks human Keep status")
        if rank in content_by_rank:
            raise LLMProtocolError(f"Duplicate approved demo rank {rank}")
        content_by_rank[rank] = expected
        approved_without_hash.append(item)
    approved_digest = sha256_text(canonical_json(approved_without_hash))
    if hashes.get("approved_case_content_sha256") != approved_digest:
        raise LLMProtocolError("Aggregate approved-demo content hash mismatch")
    membership_payload: dict[str, list[int]] = {}
    for bank_id, raw_spec in banks.items():
        if not isinstance(raw_spec, Mapping):
            raise LLMProtocolError(f"Malformed evaluation bank {bank_id}")
        ranks = [int(rank) for rank in raw_spec.get("ordered_search_ranks", [])]
        try:
            ordered_hashes = [content_by_rank[rank] for rank in ranks]
        except KeyError as exc:
            raise LLMProtocolError(
                f"Evaluation bank {bank_id} references an unapproved rank"
            ) from exc
        expected = sha256_text(
            canonical_json(
                {
                    "bank_id": bank_id,
                    "ordered_search_ranks": ranks,
                    "ordered_case_content_sha256": ordered_hashes,
                }
            )
        )
        if raw_spec.get("membership_sha256") != expected:
            raise LLMProtocolError(f"Evaluation-bank hash mismatch for {bank_id}")
        membership_payload[str(bank_id)] = ranks
    global_membership = sha256_text(canonical_json(membership_payload))
    if hashes.get("bank_membership_sha256") != global_membership:
        raise LLMProtocolError("Aggregate demo-bank membership hash mismatch")


def _demo_output_labels(output: Mapping[str, Any]) -> list[str]:
    validated = builder.validate_structured_output(output)
    return validated["acts"] + validated["means"] + validated["purposes"]


def load_demo_bank_for_setting(
    bank_id: str,
    demo_bank: Mapping[str, Any],
    config: Mapping[str, Any],
    benchmark: Mapping[int, dict[str, Any]],
    *,
    actual_test_jurisdictions: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_demo_bank_internal_hashes(demo_bank)
    try:
        bank_spec = demo_bank["evaluation_banks"][bank_id]
        config_spec = config["methods"]["M4"]["evaluation_banks"][bank_id]
    except KeyError as exc:
        raise LLMProtocolError(f"Unknown frozen demo bank {bank_id}") from exc
    ranks = [int(value) for value in bank_spec["ordered_search_ranks"]]
    if len(ranks) != 6 or len(set(ranks)) != 6:
        raise LLMProtocolError(f"{bank_id} does not contain exactly six unique demos")
    if ranks != [int(value) for value in config_spec["ordered_search_ranks"]]:
        raise LLMProtocolError(f"{bank_id} rank order differs between frozen artifacts")
    if bank_spec["membership_sha256"] != config_spec["membership_sha256"]:
        raise LLMProtocolError(f"{bank_id} membership hash mismatch")
    heldout = set(actual_test_jurisdictions)
    expected_heldout = set(bank_spec.get("heldout_test_jurisdictions", []))
    if heldout and heldout != expected_heldout:
        raise LLMProtocolError(
            f"{bank_id} actual held-out jurisdictions differ from the frozen bank"
        )
    approved = {
        int(item["search_rank"]): item for item in demo_bank.get("approved_cases", [])
    }
    demos: list[dict[str, Any]] = []
    for order, rank in enumerate(ranks, start=1):
        if rank not in approved or rank not in benchmark:
            raise LLMProtocolError(f"{bank_id} demo rank {rank} is absent")
        raw = approved[rank]
        case = benchmark[rank]
        if raw.get("human_approved") is not True or raw.get("frozen") is not True:
            raise LLMProtocolError(f"{bank_id} demo rank {rank} is not approved/frozen")
        if raw.get("jurisdiction") != case["jurisdiction"]:
            raise LLMProtocolError(f"{bank_id} demo jurisdiction mismatch for rank {rank}")
        if raw.get("canonical_url") != case["canonical_url"]:
            raise LLMProtocolError(f"{bank_id} demo URL mismatch for rank {rank}")
        if sha256_text(case["fact_summary"]) != raw.get("fact_summary_sha256"):
            raise LLMProtocolError(f"{bank_id} Fact Summary hash mismatch for rank {rank}")
        if raw.get("fact_summary") != case["fact_summary"]:
            raise LLMProtocolError(f"{bank_id} Fact Summary mismatch for rank {rank}")
        if _demo_output_labels(raw["output"]) != case["silver_reference_labels"]:
            raise LLMProtocolError(f"{bank_id} AMP output mismatch for rank {rank}")
        if case["jurisdiction"] in heldout:
            raise LLMProtocolError(
                f"{bank_id} leaks held-out jurisdiction {case['jurisdiction']}"
            )
        demos.append(
            {
                "demo_id": raw["demo_id"],
                "demo_order": order,
                "search_rank": rank,
                "canonical_url": raw["canonical_url"],
                "jurisdiction": raw["jurisdiction"],
                "fact_summary": raw["fact_summary"],
                "output": raw["output"],
                "human_approved": True,
                "frozen": True,
                "approval_record": raw["approval_record"],
            }
        )
    metadata = {
        "demo_bank_id": bank_id,
        "demo_bank_version": demo_bank["bank_version"],
        "demo_bank_membership_sha256": bank_spec["membership_sha256"],
        "global_demo_bank_membership_sha256": demo_bank["hashes"][
            "bank_membership_sha256"
        ],
        "demo_order": [
            {
                "demo_order": item["demo_order"],
                "demo_id": item["demo_id"],
                "search_rank": item["search_rank"],
                "jurisdiction": item["jurisdiction"],
            }
            for item in demos
        ],
        "heldout_test_jurisdictions": sorted(heldout),
    }
    return demos, metadata


def target_membership_hash(cases: Sequence[Mapping[str, Any]], spec: RunSpec) -> str:
    return sha256_text(
        "".join(
            f"{case['search_rank']}\t{case['canonical_url']}\t{case['role']}\t"
            f"{spec.evaluation}\t{spec.fold or ''}\n"
            for case in sorted(cases, key=lambda item: int(item["search_rank"]))
        )
    )


def _expected_builder_demo_hash(
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
) -> str | None:
    if demos is None:
        return None
    if demo_metadata is None:
        raise LLMProtocolError("M4 demo metadata is missing")
    normalized: list[dict[str, Any]] = []
    for expected_order, raw in enumerate(demos, start=1):
        if int(raw.get("demo_order") or 0) != expected_order:
            raise LLMProtocolError("M4 demonstration order changed before request build")
        normalized.append(
            {
                "demo_id": raw["demo_id"],
                "demo_order": expected_order,
                "search_rank": int(raw["search_rank"]),
                "canonical_url": raw["canonical_url"],
                "jurisdiction": raw["jurisdiction"],
                "fact_summary": raw["fact_summary"],
                "output": builder.validate_structured_output(raw["output"]),
                "human_approved": raw.get("human_approved") is True,
                "frozen": raw.get("frozen") is True,
                "approval_record": raw["approval_record"],
            }
        )
    return sha256_text(
        canonical_json(
            {
                "version": str(demo_metadata["demo_bank_version"]),
                "demos": normalized,
            }
        )
    )


def validate_builder_result(
    built: Mapping[str, Any],
    spec: RunSpec,
    case: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Bind builder payload/metadata to the already validated frozen inputs."""

    payload = built.get("payload")
    metadata = built.get("metadata")
    if not isinstance(payload, Mapping) or not isinstance(metadata, Mapping):
        raise LLMProtocolError("Request builder returned malformed payload or metadata")
    payload_sha256 = sha256_text(canonical_json(payload))
    expected_metadata = {
        "experiment_id": config["methods"][spec.method]["experiment_id"],
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "model": MODEL_ALIAS,
        "prompt_version": config["methods"][spec.method]["prompt_version"],
        "demo_bank_version": (
            str(demo_metadata["demo_bank_version"])
            if demo_metadata is not None
            else None
        ),
        "input_text_sha256": sha256_text(case["fact_summary"]),
        "shared_instruction_sha256": contract["marked_block_sha256"],
        "schema_sha256": config["structured_output"]["schema_sha256"],
        "demo_bank_sha256": _expected_builder_demo_hash(demos, demo_metadata),
        "config_sha256": contract["config_sha256"],
        "request_payload_sha256": payload_sha256,
    }
    if dict(metadata) != expected_metadata:
        mismatches = {
            key: {"expected": value, "observed": metadata.get(key)}
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        unexpected = sorted(set(metadata) - set(expected_metadata))
        missing = sorted(set(expected_metadata) - set(metadata))
        raise LLMProtocolError(
            "Request-builder metadata is not bound to the frozen case/artifacts: "
            + canonical_json(
                {"mismatches": mismatches, "unexpected": unexpected, "missing": missing}
            )
        )
    if payload.get("model") != MODEL_ALIAS:
        raise LLMProtocolError("Builder payload did not use the frozen model alias")
    if "api_key" in canonical_json(payload).lower() or _contains_secret_key(payload):
        raise LLMProtocolError("Request payload contains a credential-like field")
    return {
        "builder_payload_sha256": payload_sha256,
        "builder_metadata_sha256": sha256_text(canonical_json(metadata)),
    }


def build_request_for_case(
    spec: RunSpec,
    case: Mapping[str, Any],
    *,
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
    heldout_jurisdictions: Sequence[str],
    effective_model_id: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    m3_prompt_path: Path,
    m4_prompt_path: Path,
) -> dict[str, Any]:
    target = {
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "fact_summary": case["fact_summary"],
    }
    if spec.method == "M3":
        built = builder.build_m3_request(
            target,
            config_path=config_path,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
        )
    else:
        if demo_metadata is None:
            raise LLMProtocolError("M4 demo metadata is missing")
        built = builder.build_m4_request(
            target,
            demos,
            demo_bank_version=str(demo_metadata["demo_bank_version"]),
            heldout_jurisdictions=heldout_jurisdictions,
            config_path=config_path,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
        )
    builder_hashes = validate_builder_result(
        built,
        spec,
        case,
        contract=contract,
        config=config,
        demos=demos,
        demo_metadata=demo_metadata,
    )
    payload = dict(built["payload"])
    payload["model"] = effective_model_id
    if "api_key" in canonical_json(payload).lower() or _contains_secret_key(payload):
        raise LLMProtocolError("Request payload contains a credential-like field")
    return {
        "payload": payload,
        "request_sha256": sha256_text(canonical_json(payload)),
        "builder_metadata": dict(built["metadata"]),
        **builder_hashes,
    }


def revalidate_built_request(
    request: Mapping[str, Any],
    spec: RunSpec,
    case: Mapping[str, Any],
    *,
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
    heldout_jurisdictions: Sequence[str],
    effective_model_id: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    config_path: Path,
    m3_prompt_path: Path,
    m4_prompt_path: Path,
) -> None:
    """Rebuild after the final artifact-hash check and compare byte-level hashes."""

    payload = request.get("payload")
    metadata = request.get("builder_metadata")
    if not isinstance(payload, Mapping) or not isinstance(metadata, Mapping):
        raise LLMProtocolError("Prepared request lost its payload or builder metadata")
    current_request_sha = sha256_text(canonical_json(payload))
    current_metadata_sha = sha256_text(canonical_json(metadata))
    expected_current = {
        "request_sha256": current_request_sha,
        "builder_metadata_sha256": current_metadata_sha,
    }
    for field, observed in expected_current.items():
        if request.get(field) != observed:
            raise LLMProtocolError(
                f"Prepared request {field} changed for rank {case['search_rank']}"
            )
    fresh = build_request_for_case(
        spec,
        case,
        demos=demos,
        demo_metadata=demo_metadata,
        heldout_jurisdictions=heldout_jurisdictions,
        effective_model_id=effective_model_id,
        contract=contract,
        config=config,
        config_path=config_path,
        m3_prompt_path=m3_prompt_path,
        m4_prompt_path=m4_prompt_path,
    )
    compared_fields = (
        "request_sha256",
        "builder_payload_sha256",
        "builder_metadata_sha256",
        "builder_metadata",
        "payload",
    )
    changed = [field for field in compared_fields if request.get(field) != fresh.get(field)]
    if changed:
        raise LLMProtocolError(
            f"Request rebuild changed after canonical revalidation for rank "
            f"{case['search_rank']}: {changed}"
        )


def _object_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _listed_model_ids(client: Any) -> tuple[list[str], str | None]:
    try:
        page = client.models.list()
        ids = sorted(
            {
                str(_object_attr(item, "id"))
                for item in page
                if _object_attr(item, "id")
            }
        )
        return ids, None
    except Exception as exc:  # Access listing may be restricted independently.
        return [], f"{type(exc).__name__}: model listing unavailable"


def resolve_model_access(client: Any, *, alias: str = MODEL_ALIAS) -> dict[str, Any]:
    """Confirm model access and choose one identifier without inference."""

    ids, listing_warning = _listed_model_ids(client)
    snapshots = sorted(model_id for model_id in ids if DATED_MODEL_PATTERN.fullmatch(model_id))
    retrieved_id: str | None = None
    retrieve_warning: str | None = None
    try:
        model = client.models.retrieve(alias)
        candidate = _object_attr(model, "id")
        retrieved_id = str(candidate) if candidate else None
    except Exception as exc:
        retrieve_warning = f"{type(exc).__name__}: alias metadata retrieval unavailable"
    if snapshots:
        effective = snapshots[-1]
        selection_basis = "LATEST_EXPOSED_DATED_SNAPSHOT"
    elif retrieved_id and DATED_MODEL_PATTERN.fullmatch(retrieved_id):
        effective = retrieved_id
        selection_basis = "DATED_SNAPSHOT_RETURNED_FOR_ALIAS"
    elif retrieved_id == alias or alias in ids:
        effective = alias
        selection_basis = "FROZEN_ALIAS_ONLY"
    else:
        raise LLMProtocolError(
            "The API project did not expose or confirm access to gpt-5.6-luna"
        )
    warnings = [warning for warning in (listing_warning, retrieve_warning) if warning]
    return {
        "requested_model_id": alias,
        "effective_model_id": effective,
        "selection_basis": selection_basis,
        "returned_model_metadata_id": retrieved_id,
        "available_dated_snapshot_ids": snapshots,
        "warnings": warnings,
    }


def perform_model_access_check(
    client: Any,
    *,
    sdk_version: str,
    contract: Mapping[str, Any],
    config_path: Path,
    marker_path: Path,
    secret: str,
) -> dict[str, Any]:
    model = resolve_model_access(client)
    marker = {
        "schema_version": MODEL_ACCESS_SCHEMA_VERSION,
        "runner_version": VERSION,
        "status": "MODEL_ACCESS_CONFIRMED",
        "checked_at": utc_now(),
        "sdk_version": sdk_version,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": contract["config_sha256"],
        "schema_sha256": contract["config"]["structured_output"]["schema_sha256"],
        **model,
    }
    atomic_json(marker_path, marker, secret=secret)
    return marker


def load_model_access_marker(
    path: Path, contract: Mapping[str, Any], sdk_version: str
) -> dict[str, Any]:
    if not path.is_file():
        raise LLMProtocolError(
            "Model-access gate is absent; run --check-model-access first"
        )
    marker = load_json(path)
    if marker.get("status") != "MODEL_ACCESS_CONFIRMED":
        raise LLMProtocolError("Model-access gate did not pass")
    if marker.get("config_sha256") != contract["config_sha256"]:
        raise LLMProtocolError("Model-access gate belongs to a different frozen config")
    if marker.get("sdk_version") != sdk_version:
        raise LLMProtocolError(
            "OpenAI SDK version changed after the model-access check; rerun the check"
        )
    effective = str(marker.get("effective_model_id") or "")
    if effective != MODEL_ALIAS and not DATED_MODEL_PATTERN.fullmatch(effective):
        raise LLMProtocolError("Model-access gate contains an invalid model identifier")
    return marker


def _response_headers(exc: BaseException) -> Mapping[str, Any]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return headers if isinstance(headers, Mapping) else {}


def retry_after_seconds(exc: BaseException, *, now: datetime | None = None) -> float | None:
    headers = _response_headers(exc)
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        milliseconds = headers.get("retry-after-ms") or headers.get("Retry-After-Ms")
        if milliseconds is not None:
            try:
                return max(0.0, float(milliseconds) / 1000.0)
            except (TypeError, ValueError):
                return None
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (parsed - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def exception_status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_transient_error(exc: BaseException) -> bool:
    status = exception_status_code(exc)
    if status in {408, 409, 425, 429} or (status is not None and status >= 500):
        return True
    name = type(exc).__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def is_fatal_access_error(exc: BaseException) -> bool:
    return exception_status_code(exc) in {401, 403, 404}


def safe_error_message(exc: BaseException, secret: str) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    # Keep diagnostics useful while avoiding huge echoed payloads.
    return message[:1000]


def _response_refusal(response: Any) -> str | None:
    output = _object_attr(response, "output", []) or []
    for item in output:
        contents = _object_attr(item, "content", []) or []
        for content in contents:
            if _object_attr(content, "type") == "refusal":
                return str(_object_attr(content, "refusal", "MODEL_REFUSAL"))
    return None


def _incomplete_details(response: Any) -> dict[str, Any] | None:
    value = _object_attr(response, "incomplete_details")
    if value is None:
        return None
    try:
        primitive = safe_primitive(value)
    except LLMProtocolError:
        reason = _object_attr(value, "reason")
        return {"reason": str(reason)} if reason is not None else None
    return dict(primitive) if isinstance(primitive, Mapping) else None


def _response_attempt_provenance(
    response: Any, *, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Select auditable response metadata without serializing narrative output."""

    return {
        "response_id": (
            str(_object_attr(response, "id"))
            if _object_attr(response, "id") is not None
            else None
        ),
        "returned_model_id": (
            str(_object_attr(response, "model"))
            if _object_attr(response, "model") is not None
            else None
        ),
        "response_status": _object_attr(response, "status"),
        "incomplete_details": _incomplete_details(response),
        "token_usage": _usage(response),
        "max_output_tokens": payload.get("max_output_tokens"),
        "actual_request_sha256": sha256_text(canonical_json(payload)),
    }


def parse_response(response: Any) -> dict[str, Any]:
    status = _object_attr(response, "status")
    if status not in (None, "completed"):
        details = _incomplete_details(response)
        if (
            status == "incomplete"
            and isinstance(details, Mapping)
            and details.get("reason") == "max_output_tokens"
        ):
            raise MaxOutputTokensIncomplete(response)
        raise LLMProtocolError(f"Response status is {status!r}; details={details}")
    refusal = _response_refusal(response)
    if refusal is not None:
        raise LLMProtocolError("The API returned a model refusal")
    output_text = _object_attr(response, "output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise LLMProtocolError("The API response contains no structured output text")
    try:
        raw = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise LLMProtocolError("Structured output text is not valid JSON") from exc
    validated = builder.validate_structured_output(raw)
    return {
        "raw_structured_response_text": output_text,
        "raw_structured_response": raw,
        "validated_prediction": validated,
    }


def output_token_fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one permitted payload variant, changing only token budget."""

    if payload.get("max_output_tokens") != INITIAL_MAX_OUTPUT_TOKENS:
        raise LLMProtocolError(
            "Technical fallback requires a 512-token frozen base payload"
        )
    fallback = dict(payload)
    fallback["max_output_tokens"] = FALLBACK_MAX_OUTPUT_TOKENS
    base_without_budget = dict(payload)
    fallback_without_budget = dict(fallback)
    del base_without_budget["max_output_tokens"]
    del fallback_without_budget["max_output_tokens"]
    if canonical_json(base_without_budget) != canonical_json(fallback_without_budget):
        raise LLMProtocolError(
            "Technical fallback payload changed outside max_output_tokens"
        )
    return fallback


def _attempt_error_event(
    exc: BaseException,
    *,
    payload: Mapping[str, Any],
    attempt: int,
    phase: str,
    secret: str,
) -> dict[str, Any]:
    transient = is_transient_error(exc)
    event: dict[str, Any] = {
        "attempt": attempt,
        "timestamp": utc_now(),
        "request_phase": phase,
        "max_output_tokens": payload.get("max_output_tokens"),
        "actual_request_sha256": sha256_text(canonical_json(payload)),
        "error_type": type(exc).__name__,
        "http_status": exception_status_code(exc),
        "transient": transient,
        "message": safe_error_message(exc, secret),
    }
    if isinstance(exc, MaxOutputTokensIncomplete):
        event.update(_response_attempt_provenance(exc.response, payload=payload))
        event["technical_fallback_trigger"] = (
            phase == "INITIAL_512"
            and event.get("response_status") == "incomplete"
            and isinstance(event.get("incomplete_details"), Mapping)
            and event["incomplete_details"].get("reason") == "max_output_tokens"
        )
    return event


def invoke_with_retries(
    client: Any,
    payload: Mapping[str, Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    secret: str,
    start_with_fallback: bool = False,
    prior_fallback_attempts: int = 0,
    prior_primary_incomplete_provenance: Mapping[str, Any] | None = None,
    primary_attempt_limit: int | None = None,
    fallback_attempt_ceiling: int = MAX_FALLBACK_ATTEMPTS_PER_CASE,
    technical_exception_id: str | None = None,
    technical_exception_sha256: str | None = None,
    fallback_attempt_reserver: (
        Callable[[int, Mapping[str, Any]], None] | None
    ) = None,
    primary_attempt_reserver: (
        Callable[[int, Mapping[str, Any]], None] | None
    ) = None,
) -> dict[str, Any]:
    """Invoke a case under the frozen base and additive fallback policies.

    Normal HTTP/transient retries retain their prior behavior at 512.  The
    2048-token payload is reachable only from an exact max-output incomplete or
    an independently validated persisted instance of that trigger.  Every
    actual 2048-token call counts toward a cumulative two-call ceiling.
    """

    if max_attempts < 1:
        raise LLMProtocolError("max_attempts must be positive")
    if primary_attempt_limit is not None and (
        isinstance(primary_attempt_limit, bool)
        or not isinstance(primary_attempt_limit, int)
        or primary_attempt_limit < 1
    ):
        raise LLMProtocolError("primary_attempt_limit must be a positive integer")
    if payload.get("max_output_tokens") != INITIAL_MAX_OUTPUT_TOKENS:
        raise LLMProtocolError("Initial request must retain max_output_tokens=512")
    if (
        isinstance(prior_fallback_attempts, bool)
        or not isinstance(prior_fallback_attempts, int)
        or prior_fallback_attempts < 0
        or prior_fallback_attempts > fallback_attempt_ceiling
    ):
        raise LLMProtocolError("Invalid cumulative fallback-attempt count")
    exception_active = (
        fallback_attempt_ceiling == RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS
        and technical_exception_id == RANK_1340_EXCEPTION_ID
        and technical_exception_sha256
        == EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
        and start_with_fallback
        and prior_fallback_attempts == MAX_FALLBACK_ATTEMPTS_PER_CASE
    )
    if fallback_attempt_ceiling != MAX_FALLBACK_ATTEMPTS_PER_CASE and not exception_active:
        raise LLMProtocolError("Invalid rank-1340 exception execution policy")
    if start_with_fallback:
        proof = prior_primary_incomplete_provenance
        base_sha256 = sha256_text(canonical_json(payload))
        if not (
            isinstance(proof, Mapping)
            and proof.get("response_status") == "incomplete"
            and isinstance(proof.get("incomplete_details"), Mapping)
            and proof["incomplete_details"].get("reason") == "max_output_tokens"
            and proof.get("max_output_tokens") == INITIAL_MAX_OUTPUT_TOKENS
            and proof.get("actual_request_sha256") == base_sha256
        ):
            raise LLMProtocolError(
                "Direct fallback resume lacks an exact persisted 512 trigger"
            )
    if start_with_fallback and prior_fallback_attempts >= fallback_attempt_ceiling:
        raise LLMProtocolError("The per-case 2048-token fallback ceiling is exhausted")

    started = clock()
    retry_events: list[dict[str, Any]] = []
    actual_calls = 0
    fallback_calls = 0
    initial_incomplete_provenance = (
        dict(prior_primary_incomplete_provenance)
        if prior_primary_incomplete_provenance is not None
        else None
    )
    base_request_sha256 = sha256_text(canonical_json(payload))

    def result_metadata(actual_payload: Mapping[str, Any]) -> dict[str, Any]:
        metadata = {
            "latency_seconds": max(0.0, clock() - started),
            "retry_count": max(0, actual_calls - 1),
            "request_attempt_count": actual_calls,
            "retry_events": retry_events,
            "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
            "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
            "base_request_sha256": base_request_sha256,
            "actual_request_sha256": sha256_text(canonical_json(actual_payload)),
            "initial_max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
            "effective_max_output_tokens": actual_payload.get("max_output_tokens"),
            "output_token_fallback_used": fallback_calls > 0,
            "output_token_fallback_attempts_this_invocation": fallback_calls,
            "prior_output_token_fallback_attempts": prior_fallback_attempts,
            "cumulative_output_token_fallback_attempts": (
                prior_fallback_attempts + fallback_calls
            ),
            "initial_incomplete_response_provenance": initial_incomplete_provenance,
        }
        if exception_active:
            metadata.update(
                {
                    "technical_exception_id": RANK_1340_EXCEPTION_ID,
                    "technical_exception_sha256": (
                        EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
                    ),
                }
            )
        return metadata

    if not start_with_fallback:
        allowed_primary_attempts = min(
            max_attempts,
            primary_attempt_limit if primary_attempt_limit is not None else max_attempts,
        )
        for primary_attempt in range(1, allowed_primary_attempts + 1):
            if primary_attempt_reserver is not None:
                primary_attempt_reserver(primary_attempt, payload)
            actual_calls += 1
            try:
                response = client.responses.create(**dict(payload))
                parsed = parse_response(response)
                return {
                    "ok": True,
                    "response": response,
                    "parsed": parsed,
                    **result_metadata(payload),
                }
            except Exception as exc:
                event = _attempt_error_event(
                    exc,
                    payload=payload,
                    attempt=actual_calls,
                    phase="INITIAL_512",
                    secret=secret,
                )
                if isinstance(exc, MaxOutputTokensIncomplete):
                    retry_events.append(event)
                    initial_incomplete_provenance = {
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
                if event["transient"] and primary_attempt < allowed_primary_attempts:
                    advised = retry_after_seconds(exc)
                    exponential = min(
                        MAX_BACKOFF_SECONDS,
                        base_backoff_seconds * (2 ** (primary_attempt - 1)),
                    )
                    delay = min(MAX_BACKOFF_SECONDS, max(exponential, advised or 0.0))
                    event["retry_after_seconds"] = advised
                    event["sleep_seconds"] = delay
                    retry_events.append(event)
                    sleeper(delay)
                    continue
                retry_events.append(event)
                return {
                    "ok": False,
                    "error": event,
                    "fatal_access_error": is_fatal_access_error(exc),
                    **result_metadata(payload),
                }

    fallback_payload = output_token_fallback_payload(payload)
    remaining_fallback_attempts = (
        fallback_attempt_ceiling - prior_fallback_attempts
    )
    for fallback_attempt in range(1, remaining_fallback_attempts + 1):
        fallback_calls += 1
        if fallback_attempt_reserver is not None:
            fallback_attempt_reserver(
                prior_fallback_attempts + fallback_calls, fallback_payload
            )
        actual_calls += 1
        try:
            response = client.responses.create(**dict(fallback_payload))
            parsed = parse_response(response)
            return {
                "ok": True,
                "response": response,
                "parsed": parsed,
                **result_metadata(fallback_payload),
            }
        except Exception as exc:
            event = _attempt_error_event(
                exc,
                payload=fallback_payload,
                attempt=actual_calls,
                phase="FALLBACK_2048",
                secret=secret,
            )
            retryable_fallback = isinstance(exc, MaxOutputTokensIncomplete) or bool(
                event["transient"]
            )
            if retryable_fallback and fallback_attempt < remaining_fallback_attempts:
                advised = retry_after_seconds(exc)
                exponential = min(
                    MAX_BACKOFF_SECONDS,
                    base_backoff_seconds * (2 ** (fallback_attempt - 1)),
                )
                delay = min(MAX_BACKOFF_SECONDS, max(exponential, advised or 0.0))
                event["retry_after_seconds"] = advised
                event["sleep_seconds"] = delay
                retry_events.append(event)
                sleeper(delay)
                continue
            retry_events.append(event)
            return {
                "ok": False,
                "error": event,
                "fatal_access_error": is_fatal_access_error(exc),
                **result_metadata(fallback_payload),
            }
    raise AssertionError("unreachable")


def _usage(response: Any) -> dict[str, Any] | None:
    usage = _object_attr(response, "usage")
    if usage is None:
        return None
    value = safe_primitive(usage)
    return value if isinstance(value, dict) else {"value": value}


def _validated_technical_execution_provenance(
    result: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "technical_amendment_id",
        "technical_amendment_sha256",
        "base_request_sha256",
        "actual_request_sha256",
        "initial_max_output_tokens",
        "effective_max_output_tokens",
        "output_token_fallback_used",
        "output_token_fallback_attempts_this_invocation",
        "prior_output_token_fallback_attempts",
        "cumulative_output_token_fallback_attempts",
        "request_attempt_count",
        "initial_incomplete_response_provenance",
    )
    provenance = {field: result.get(field) for field in fields}
    exception_active = (
        result.get("technical_exception_id") == RANK_1340_EXCEPTION_ID
        and result.get("technical_exception_sha256")
        == EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
    )
    if exception_active:
        provenance.update(
            {
                "technical_exception_id": RANK_1340_EXCEPTION_ID,
                "technical_exception_sha256": (
                    EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
                ),
            }
        )
    expected = {
        "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
        "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "base_request_sha256": request["request_sha256"],
        "initial_max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
    }
    mismatches = {
        field: {"expected": value, "observed": provenance.get(field)}
        for field, value in expected.items()
        if provenance.get(field) != value
    }
    fallback_used = provenance["output_token_fallback_used"] is True
    expected_budget = (
        FALLBACK_MAX_OUTPUT_TOKENS if fallback_used else INITIAL_MAX_OUTPUT_TOKENS
    )
    if provenance["effective_max_output_tokens"] != expected_budget:
        mismatches["effective_max_output_tokens"] = {
            "expected": expected_budget,
            "observed": provenance["effective_max_output_tokens"],
        }
    request_payload = request.get("payload")
    if isinstance(request_payload, Mapping):
        expected_actual_payload = (
            output_token_fallback_payload(request_payload)
            if fallback_used
            else dict(request_payload)
        )
        expected_actual_sha = sha256_text(canonical_json(expected_actual_payload))
        if provenance["actual_request_sha256"] != expected_actual_sha:
            mismatches["actual_request_sha256"] = {
                "expected": expected_actual_sha,
                "observed": provenance["actual_request_sha256"],
            }
    elif not fallback_used and (
        provenance["actual_request_sha256"] != request["request_sha256"]
    ):
        mismatches["actual_request_sha256"] = {
            "expected": request["request_sha256"],
            "observed": provenance["actual_request_sha256"],
        }
    current = provenance["output_token_fallback_attempts_this_invocation"]
    prior = provenance["prior_output_token_fallback_attempts"]
    cumulative = provenance["cumulative_output_token_fallback_attempts"]
    if (
        isinstance(current, bool)
        or not isinstance(current, int)
        or isinstance(prior, bool)
        or not isinstance(prior, int)
        or isinstance(cumulative, bool)
        or not isinstance(cumulative, int)
        or cumulative != current + prior
        or cumulative
        > (
            RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS
            if exception_active
            else MAX_FALLBACK_ATTEMPTS_PER_CASE
        )
        or (exception_active and prior != MAX_FALLBACK_ATTEMPTS_PER_CASE)
        or (fallback_used != (current > 0))
    ):
        mismatches["fallback_attempt_counts"] = {
            "current": current,
            "prior": prior,
            "cumulative": cumulative,
        }
    proof = provenance["initial_incomplete_response_provenance"]
    if fallback_used:
        if not (
            isinstance(proof, Mapping)
            and proof.get("response_status") == "incomplete"
            and isinstance(proof.get("incomplete_details"), Mapping)
            and proof["incomplete_details"].get("reason") == "max_output_tokens"
            and proof.get("max_output_tokens") == INITIAL_MAX_OUTPUT_TOKENS
            and proof.get("actual_request_sha256") == request["request_sha256"]
        ):
            mismatches["initial_incomplete_response_provenance"] = {
                "expected": "exact persisted 512 max_output_tokens trigger",
                "observed": proof,
            }
    elif proof is not None:
        mismatches["initial_incomplete_response_provenance"] = {
            "expected": None,
            "observed": proof,
        }
    if mismatches:
        raise LLMProtocolError(
            "Invalid technical-amendment execution provenance: "
            + canonical_json(mismatches)
        )
    return provenance


def make_success_record(
    spec: RunSpec,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
    sdk_version: str,
    demo_metadata: Mapping[str, Any] | None,
    split_membership_sha256: str,
) -> dict[str, Any]:
    parsed = result["parsed"]
    response = result["response"]
    prediction = parsed["validated_prediction"]
    predicted_labels = prediction["acts"] + prediction["means"] + prediction["purposes"]
    method_config = config["methods"][spec.method]
    response_id = _object_attr(response, "id")
    returned_model = _object_attr(response, "model")
    technical_execution = _validated_technical_execution_provenance(result, request)
    _validate_exception_record_scope(spec, case, technical_execution)
    execution_timestamp = utc_now()
    run_id = sha256_text(
        canonical_json(
            {
                "method": spec.method,
                "evaluation": spec.evaluation,
                "fold": spec.fold,
                "config_sha256": contract["config_sha256"],
                "split_membership_sha256": split_membership_sha256,
                "effective_model_id": model_marker["effective_model_id"],
                "demo_bank_membership_sha256": (
                    demo_metadata or {}
                ).get("demo_bank_membership_sha256"),
            }
        )
    )[:24]
    record = {
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "execution_schema_version": EXECUTION_SCHEMA_VERSION,
        "runner_version": VERSION,
        "run_id": run_id,
        "experiment_id": method_config["experiment_id"],
        "method": spec.method,
        "method_id": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "split_or_fold": spec.split_or_fold,
        "split": "DRY_RUN_NON_TEST" if spec.dry_run else "TEST",
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "case_title": case["case_title"],
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "fact_summary": case["fact_summary"],
        "input_sha256": sha256_text(case["fact_summary"]),
        "silver_reference_terminology": "SILVER_REFERENCE_LEGACY_KEYWORDS",
        "silver_reference_labels": case["silver_reference_labels"],
        "predicted_labels": predicted_labels,
        "normalized_prediction": prediction,
        "validated_prediction": prediction,
        "raw_structured_response": parsed["raw_structured_response"],
        "raw_structured_response_text": parsed["raw_structured_response_text"],
        "label_array_canonicalization_applied": (
            parsed["raw_structured_response"] != prediction
        ),
        "requested_model_id": MODEL_ALIAS,
        "effective_requested_model_id": model_marker["effective_model_id"],
        "returned_model_id": str(returned_model) if returned_model else None,
        "execution_timestamp": execution_timestamp,
        "sdk_version": sdk_version,
        "prompt_version": method_config["prompt_version"],
        "prompt_sha256": method_config["prompt_sha256"],
        "shared_instruction_sha256": contract["marked_block_sha256"],
        "schema_sha256": config["structured_output"]["schema_sha256"],
        "demo_bank_id": (demo_metadata or {}).get("demo_bank_id"),
        "demo_bank_version": (demo_metadata or {}).get("demo_bank_version"),
        "demo_bank_membership_sha256": (
            demo_metadata or {}
        ).get("demo_bank_membership_sha256"),
        "global_demo_bank_membership_sha256": (
            demo_metadata or {}
        ).get("global_demo_bank_membership_sha256"),
        "demo_order": (demo_metadata or {}).get("demo_order", []),
        "request_sha256": request["request_sha256"],
        "request_payload_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "builder_metadata": request["builder_metadata"],
        "response_id": str(response_id) if response_id else None,
        "token_usage": _usage(response),
        "latency_seconds": float(result["latency_seconds"]),
        "retry_count": int(result["retry_count"]),
        "retry_events": result["retry_events"],
        "technical_execution": technical_execution,
        "status": "SUCCESS_VALIDATED",
        "truncated_input": False,
        "primary_cohort_id": EXPECTED_COHORT_ID,
        "config_sha256": contract["config_sha256"],
        "split_membership_sha256": split_membership_sha256,
        "test_labels_used_for_request_or_tuning": False,
    }
    return record


def make_failure_record(
    spec: RunSpec,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
    sdk_version: str,
    demo_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    technical_execution = _validated_technical_execution_provenance(result, request)
    _validate_exception_record_scope(spec, case, technical_execution)
    return {
        "failure_schema_version": FAILURE_SCHEMA_VERSION,
        "runner_version": VERSION,
        "status": "FAILED_NO_PREDICTION",
        "recorded_at": utc_now(),
        "experiment_id": config["methods"][spec.method]["experiment_id"],
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "split_or_fold": spec.split_or_fold,
        "case_id": case["case_id"],
        "search_rank": int(case["search_rank"]),
        "canonical_url": case["canonical_url"],
        "jurisdiction": case["jurisdiction"],
        "input_sha256": sha256_text(case["fact_summary"]),
        "requested_model_id": MODEL_ALIAS,
        "effective_requested_model_id": model_marker["effective_model_id"],
        "sdk_version": sdk_version,
        "prompt_version": config["methods"][spec.method]["prompt_version"],
        "prompt_sha256": config["methods"][spec.method]["prompt_sha256"],
        "schema_sha256": config["structured_output"]["schema_sha256"],
        "demo_bank_id": (demo_metadata or {}).get("demo_bank_id"),
        "demo_bank_membership_sha256": (
            demo_metadata or {}
        ).get("demo_bank_membership_sha256"),
        "request_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "builder_metadata": request["builder_metadata"],
        "latency_seconds": float(result["latency_seconds"]),
        "retry_count": int(result["retry_count"]),
        "retry_events": result["retry_events"],
        "technical_execution": technical_execution,
        "error": result["error"],
        "fatal_access_error": bool(result.get("fatal_access_error")),
        "validated_prediction": None,
        "config_sha256": contract["config_sha256"],
    }


def _state_path(directory: Path, search_rank: int) -> Path:
    return directory / f"{search_rank:06d}.json"


def validate_primary_recovery_reservation(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
) -> bool:
    if not path.is_file():
        return False
    document = load_json(path)
    expected = {
        "schema_version": PRIMARY_RECOVERY_RESERVATION_SCHEMA_VERSION,
        "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
        "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "status": "PRIMARY_RECOVERY_CALL_DURABLY_RESERVED",
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "search_rank": int(case["search_rank"]),
        "base_request_sha256": request["request_sha256"],
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
    }
    mismatches = {
        field: {"expected": value, "observed": document.get(field)}
        for field, value in expected.items()
        if document.get(field) != value
    }
    if mismatches:
        raise LLMProtocolError(
            f"Invalid primary-recovery reservation for rank {case['search_rank']}: "
            + canonical_json(mismatches)
        )
    return True


def reserve_primary_recovery_attempt(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
    primary_attempt_number: int,
    actual_payload: Mapping[str, Any],
    secret: str,
) -> None:
    """Durably reserve rank 551's sole additional 512 call before sending."""

    if primary_attempt_number != 1:
        raise LLMProtocolError("Legacy primary recovery permits exactly one call")
    if validate_primary_recovery_reservation(
        path, case=case, request=request, spec=spec
    ):
        raise LLMProtocolError(
            f"Rank {case['search_rank']} already reserved its sole 512 recovery call"
        )
    if canonical_json(actual_payload) != canonical_json(request["payload"]):
        raise LLMProtocolError(
            f"Primary-recovery payload drift for rank {case['search_rank']}"
        )
    atomic_json(
        path,
        {
            "schema_version": PRIMARY_RECOVERY_RESERVATION_SCHEMA_VERSION,
            "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
            "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
            "status": "PRIMARY_RECOVERY_CALL_DURABLY_RESERVED",
            "reserved_at": utc_now(),
            "method": spec.method,
            "evaluation": spec.evaluation,
            "fold": spec.fold,
            "search_rank": int(case["search_rank"]),
            "base_request_sha256": request["request_sha256"],
            "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
            "actual_request_sha256": sha256_text(canonical_json(actual_payload)),
        },
        secret=secret,
    )


def _fallback_reservation_count(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
) -> int:
    if not path.is_file():
        return 0
    document = load_json(path)
    expected = {
        "schema_version": FALLBACK_RESERVATION_SCHEMA_VERSION,
        "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
        "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "search_rank": int(case["search_rank"]),
        "base_request_sha256": request["request_sha256"],
    }
    mismatches = {
        field: {"expected": value, "observed": document.get(field)}
        for field, value in expected.items()
        if document.get(field) != value
    }
    reservations = document.get("reservations")
    if not isinstance(reservations, list):
        mismatches["reservations"] = {"expected": "list", "observed": type(reservations).__name__}
        reservations = []
    exception_active = (
        _is_rank_1340_exception_scope(spec, case)
        and document.get("technical_exception_id") == RANK_1340_EXCEPTION_ID
        and document.get("technical_exception_sha256")
        == EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
    )
    if (
        document.get("technical_exception_id") is not None
        or document.get("technical_exception_sha256") is not None
    ) and not exception_active:
        mismatches["technical_exception"] = {
            "expected": "exact M4 A2 Fold 1 rank-1340 exception",
            "observed": document.get("technical_exception_id"),
        }
    expected_fallback_sha = sha256_text(
        canonical_json(output_token_fallback_payload(request["payload"]))
    )
    for index, reservation in enumerate(reservations, start=1):
        if not (
            isinstance(reservation, Mapping)
            and reservation.get("fallback_attempt_number") == index
            and reservation.get("max_output_tokens") == FALLBACK_MAX_OUTPUT_TOKENS
            and reservation.get("actual_request_sha256") == expected_fallback_sha
        ):
            mismatches[f"reservation_{index}"] = {
                "expected": "sequential, 2048, canonical fallback hash",
                "observed": safe_primitive(reservation),
            }
    ceiling = (
        RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS
        if exception_active
        else MAX_FALLBACK_ATTEMPTS_PER_CASE
    )
    if len(reservations) > ceiling:
        mismatches["reservation_count"] = {
            "expected_maximum": ceiling,
            "observed": len(reservations),
        }
    if document.get("reserved_fallback_attempts") != len(reservations):
        mismatches["reserved_fallback_attempts"] = {
            "expected": len(reservations),
            "observed": document.get("reserved_fallback_attempts"),
        }
    if mismatches:
        raise LLMProtocolError(
            f"Invalid fallback reservation journal for rank {case['search_rank']}: "
            + canonical_json(mismatches)
        )
    return len(reservations)


def reserve_fallback_attempt(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
    fallback_attempt_number: int,
    actual_payload: Mapping[str, Any],
    secret: str,
    technical_exception_id: str | None = None,
    technical_exception_sha256: str | None = None,
) -> None:
    """Durably reserve a 2048 call before it can reach the API client."""

    existing_count = _fallback_reservation_count(
        path, case=case, request=request, spec=spec
    )
    exception_active = (
        _is_rank_1340_exception_scope(spec, case)
        and technical_exception_id == RANK_1340_EXCEPTION_ID
        and technical_exception_sha256
        == EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
    )
    ceiling = (
        RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS
        if exception_active
        else MAX_FALLBACK_ATTEMPTS_PER_CASE
    )
    if (
        fallback_attempt_number != existing_count + 1
        or fallback_attempt_number > ceiling
    ):
        raise LLMProtocolError(
            f"Fallback reservation sequence/ceiling violation for rank "
            f"{case['search_rank']}"
        )
    expected_payload = output_token_fallback_payload(request["payload"])
    if canonical_json(actual_payload) != canonical_json(expected_payload):
        raise LLMProtocolError(
            f"Fallback reservation payload drift for rank {case['search_rank']}"
        )
    reservations: list[dict[str, Any]] = []
    created_at = utc_now()
    if path.is_file():
        existing = load_json(path)
        reservations = [dict(item) for item in existing["reservations"]]
        created_at = str(existing.get("created_at") or created_at)
    reservations.append(
        {
            "fallback_attempt_number": fallback_attempt_number,
            "reserved_at": utc_now(),
            "max_output_tokens": FALLBACK_MAX_OUTPUT_TOKENS,
            "actual_request_sha256": sha256_text(canonical_json(actual_payload)),
        }
    )
    document = {
            "schema_version": FALLBACK_RESERVATION_SCHEMA_VERSION,
            "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
            "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
            "status": "FALLBACK_CALLS_DURABLY_RESERVED",
            "created_at": created_at,
            "updated_at": utc_now(),
            "method": spec.method,
            "evaluation": spec.evaluation,
            "fold": spec.fold,
            "search_rank": int(case["search_rank"]),
            "base_request_sha256": request["request_sha256"],
            "reserved_fallback_attempts": len(reservations),
            "reservations": reservations,
        }
    if exception_active:
        document.update(
            {
                "technical_exception_id": RANK_1340_EXCEPTION_ID,
                "technical_exception_sha256": (
                    EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
                ),
            }
        )
    atomic_json(path, document, secret=secret)


LEGACY_MAX_OUTPUT_INCOMPLETE_MESSAGE = (
    "Response status is 'incomplete'; "
    "details={'reason': 'max_output_tokens'}"
)
LEGACY_LABEL_ORDER_MESSAGE = "means labels are not in frozen ontology order"


def _exact_primary_incomplete_provenance(
    failure: Mapping[str, Any], request: Mapping[str, Any], *, allow_legacy: bool
) -> dict[str, Any] | None:
    """Recover a fail-closed 512-token trigger proof from failure history."""

    if failure.get("request_sha256") != request.get("request_sha256"):
        return None
    technical = failure.get("technical_execution")
    if isinstance(technical, Mapping):
        proof = technical.get("initial_incomplete_response_provenance")
        if (
            isinstance(proof, Mapping)
            and proof.get("response_status") == "incomplete"
            and isinstance(proof.get("incomplete_details"), Mapping)
            and proof["incomplete_details"].get("reason") == "max_output_tokens"
            and proof.get("max_output_tokens") == INITIAL_MAX_OUTPUT_TOKENS
            and proof.get("actual_request_sha256") == request.get("request_sha256")
        ):
            return dict(proof)
    if not allow_legacy:
        return None
    error = failure.get("error")
    if not isinstance(error, Mapping):
        return None
    if (
        error.get("error_type") != "LLMProtocolError"
        or error.get("http_status") is not None
        or error.get("transient") is not False
        or error.get("message") != LEGACY_MAX_OUTPUT_INCOMPLETE_MESSAGE
    ):
        return None
    return {
        "source": "PRE_AMENDMENT_PERSISTED_FAILURE_HISTORY",
        "recorded_at": failure.get("recorded_at"),
        "response_status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
        "actual_request_sha256": request.get("request_sha256"),
    }


def resolve_failure_resume_policy(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
    fallback_reservation_path: Path | None = None,
    primary_recovery_reservation_path: Path | None = None,
) -> dict[str, Any]:
    """Determine whether a persisted exact trigger permits direct 2048 resume."""

    default = {
        "start_with_fallback": False,
        "prior_fallback_attempts": 0,
        "prior_primary_incomplete_provenance": None,
        "primary_attempt_limit": None,
        "primary_recovery_required": False,
        "fallback_attempt_ceiling": MAX_FALLBACK_ATTEMPTS_PER_CASE,
        "technical_exception_id": None,
        "technical_exception_sha256": None,
        "resume_source": "NO_QUALIFYING_PERSISTED_TRIGGER",
    }
    if not path.is_file():
        return default
    document = load_json(path)
    if (
        document.get("status") != "UNRESOLVED_FAILURE_HISTORY"
        or int(document.get("search_rank") or 0) != int(case["search_rank"])
    ):
        raise LLMProtocolError(
            f"Malformed failure history for rank {case['search_rank']}"
        )
    raw_history = document.get("attempt_history")
    if not isinstance(raw_history, list) or not raw_history:
        raise LLMProtocolError(
            f"Empty failure history for rank {case['search_rank']}"
        )
    history = [item for item in raw_history if isinstance(item, Mapping)]
    if len(history) != len(raw_history):
        raise LLMProtocolError(
            f"Non-object failure history entry for rank {case['search_rank']}"
        )
    expected_identity = {
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "search_rank": int(case["search_rank"]),
        "request_sha256": request["request_sha256"],
    }
    for failure in history:
        mismatches = [
            field
            for field, expected in expected_identity.items()
            if failure.get(field) != expected
        ]
        if mismatches:
            raise LLMProtocolError(
                f"Failure history identity drift for rank {case['search_rank']}: "
                f"{mismatches}"
            )

    failure_history_fallback_attempts = 0
    for failure in history:
        technical = failure.get("technical_execution")
        if not isinstance(technical, Mapping):
            continue
        count = technical.get("cumulative_output_token_fallback_attempts")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise LLMProtocolError(
                f"Invalid fallback count in history for rank {case['search_rank']}"
            )
        failure_history_fallback_attempts = max(
            failure_history_fallback_attempts, count
        )
    reserved_fallback_attempts = (
        _fallback_reservation_count(
            fallback_reservation_path,
            case=case,
            request=request,
            spec=spec,
        )
        if fallback_reservation_path is not None
        else 0
    )
    if reserved_fallback_attempts < failure_history_fallback_attempts:
        raise LLMProtocolError(
            f"Rank {case['search_rank']} failure history exceeds its durable "
            "fallback reservation journal"
        )
    prior_fallback_attempts = max(
        failure_history_fallback_attempts, reserved_fallback_attempts
    )

    rank = int(case["search_rank"])
    legacy_allowed = (
        spec.method == "M3"
        and spec.evaluation == "A1"
        and spec.fold is None
        and rank in {266, 1356}
    )
    proof: dict[str, Any] | None = None
    for failure in reversed(history):
        proof = _exact_primary_incomplete_provenance(
            failure, request, allow_legacy=legacy_allowed
        )
        if proof is not None:
            break
    if proof is None:
        if prior_fallback_attempts:
            raise LLMProtocolError(
                f"Rank {rank} has fallback calls but no valid persisted 512 trigger"
            )
        rank_551_order_only = (
            spec.method == "M3"
            and spec.evaluation == "A1"
            and spec.fold is None
            and rank == 551
            and all(
                isinstance(failure.get("error"), Mapping)
                and failure["error"].get("error_type") == "RequestBuildError"
                and failure["error"].get("http_status") is None
                and failure["error"].get("transient") is False
                and failure["error"].get("message") == LEGACY_LABEL_ORDER_MESSAGE
                for failure in history
            )
        )
        if rank_551_order_only:
            if (
                primary_recovery_reservation_path is not None
                and validate_primary_recovery_reservation(
                    primary_recovery_reservation_path,
                    case=case,
                    request=request,
                    spec=spec,
                )
            ):
                raise LLMProtocolError(
                    "Rank 551 already reserved its sole additional 512 request; "
                    "a crash-safe resume cannot send a second one"
                )
            return {
                **default,
                "primary_attempt_limit": 1,
                "primary_recovery_required": True,
                "resume_source": "PRE_AMENDMENT_LABEL_ORDER_FAILURE_HISTORY",
            }
        return default
    if _is_rank_1340_exception_scope(spec, case):
        fallback_sha = sha256_text(
            canonical_json(output_token_fallback_payload(request["payload"]))
        )
        fallback_events = [
            event
            for failure in history
            for event in failure.get("retry_events", [])
            if isinstance(event, Mapping)
            and event.get("request_phase") == "FALLBACK_2048"
        ]
        no_model_response_fields = (
            "response_id",
            "returned_model_id",
            "response_status",
            "incomplete_details",
            "token_usage",
        )
        qualifying_429_history = (
            failure_history_fallback_attempts == MAX_FALLBACK_ATTEMPTS_PER_CASE
            and reserved_fallback_attempts == MAX_FALLBACK_ATTEMPTS_PER_CASE
            and len(fallback_events) == MAX_FALLBACK_ATTEMPTS_PER_CASE
            and all(
                event.get("http_status") == 429
                and event.get("max_output_tokens") == FALLBACK_MAX_OUTPUT_TOKENS
                and event.get("actual_request_sha256") == fallback_sha
                and all(event.get(field) is None for field in no_model_response_fields)
                for event in fallback_events
            )
        )
        if qualifying_429_history:
            return {
                **default,
                "start_with_fallback": True,
                "prior_fallback_attempts": MAX_FALLBACK_ATTEMPTS_PER_CASE,
                "prior_primary_incomplete_provenance": proof,
                "fallback_attempt_ceiling": (
                    RANK_1340_EXCEPTION_MAX_FALLBACK_ATTEMPTS
                ),
                "technical_exception_id": RANK_1340_EXCEPTION_ID,
                "technical_exception_sha256": (
                    EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
                ),
                "resume_source": "AUTHORIZED_RANK_1340_HTTP_429_EXCEPTION",
            }
    if prior_fallback_attempts >= MAX_FALLBACK_ATTEMPTS_PER_CASE:
        raise LLMProtocolError(
            f"Rank {rank} exhausted its two-call 2048-token fallback ceiling"
        )
    if prior_fallback_attempts and (
        reserved_fallback_attempts <= failure_history_fallback_attempts
    ):
        # A fully recorded one-call fallback failure is non-retryable.  Only a
        # reservation not yet reflected in a result record indicates a crash
        # boundary at which the remaining allowance may be used safely.
        raise LLMProtocolError(
            f"Rank {rank} has a non-resumable prior 2048-token failure"
        )
    return {
        "start_with_fallback": True,
        "prior_fallback_attempts": prior_fallback_attempts,
        "prior_primary_incomplete_provenance": proof,
        "primary_attempt_limit": None,
        "primary_recovery_required": False,
        "resume_source": (
            "CRASH_SAFE_FALLBACK_RESERVATION"
            if reserved_fallback_attempts > failure_history_fallback_attempts
            else proof.get("source", "STRUCTURED_FAILURE_PROVENANCE")
        ),
    }


def validate_canonical_m3_a1_amendment_scope(
    spec: RunSpec,
    pending: Sequence[
        tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]
    ],
) -> None:
    """Fail closed if canonical M3 A1 would resend anything beyond 3 misses."""

    canonical_output = DEFAULT_PREDICTION_ROOT / "m3/a1_test_predictions.jsonl"
    canonical_state = DEFAULT_LOG_ROOT / "state/m3/a1"
    if not (
        spec.method == "M3"
        and spec.evaluation == "A1"
        and spec.fold is None
        and not spec.dry_run
        and spec.output_path.resolve() == canonical_output.resolve()
        and spec.state_dir.resolve() == canonical_state.resolve()
    ):
        return
    policies = {
        int(case["search_rank"]): policy for case, _request, policy in pending
    }
    pending_ranks = set(policies)
    if not pending_ranks <= EXPECTED_M3_A1_AMENDMENT_PENDING_RANKS:
        raise LLMProtocolError(
            "Amended M3 A1 resume would send non-authorized ranks: "
            + canonical_json(sorted(pending_ranks - EXPECTED_M3_A1_AMENDMENT_PENDING_RANKS))
        )
    for rank in pending_ranks & {266, 1356}:
        if policies[rank].get("start_with_fallback") is not True:
            raise LLMProtocolError(
                f"Amended M3 A1 rank {rank} must resume directly at 2048"
            )
    if 551 in pending_ranks and (
        policies[551].get("start_with_fallback") is not False
        or policies[551].get("primary_attempt_limit") != 1
        or policies[551].get("primary_recovery_required") is not True
    ):
        raise LLMProtocolError(
            "Amended M3 A1 rank 551 must receive exactly one base-512 attempt"
        )


def validate_persisted_success_record(
    record: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None,
    context: str,
) -> dict[str, Any]:
    """Validate both immutable v1.1 successes and amended v1.2 successes."""

    prediction = builder.validate_structured_output(record.get("validated_prediction"))
    labels = prediction["acts"] + prediction["means"] + prediction["purposes"]
    if (
        record.get("normalized_prediction") != prediction
        or record.get("predicted_labels") != labels
    ):
        raise LLMProtocolError(f"{context} normalized prediction fields disagree")
    technical = record.get("technical_execution")
    if technical is None and record.get("runner_version") == "1.1.0":
        return prediction
    if record.get("runner_version") != VERSION or not isinstance(
        technical, Mapping
    ):
        raise LLMProtocolError(f"{context} has an unknown execution-policy version")
    if technical.get("technical_exception_id") is not None and not (
        record.get("method") == "M4"
        and record.get("evaluation") == "A2"
        and record.get("fold") == 1
        and record.get("search_rank") == 1340
    ):
        raise LLMProtocolError(f"{context} has an out-of-scope technical exception")
    base_request = (
        request
        if request is not None
        else {"request_sha256": record.get("request_sha256")}
    )
    _validated_technical_execution_provenance(technical, base_request)
    raw = record.get("raw_structured_response")
    canonical_raw = builder.validate_structured_output(raw)
    if canonical_raw != prediction:
        raise LLMProtocolError(f"{context} raw-to-canonical prediction mismatch")
    raw_text = record.get("raw_structured_response_text")
    try:
        parsed_text = json.loads(raw_text) if isinstance(raw_text, str) else None
    except json.JSONDecodeError as exc:
        raise LLMProtocolError(f"{context} raw response text is invalid JSON") from exc
    if parsed_text != raw:
        raise LLMProtocolError(f"{context} raw response text/object mismatch")
    expected_flag = raw != prediction
    if record.get("label_array_canonicalization_applied") is not expected_flag:
        raise LLMProtocolError(f"{context} canonicalization flag is invalid")
    return prediction


def validate_success_fallback_journal(
    record: Mapping[str, Any], *, path: Path, context: str
) -> None:
    technical = record.get("technical_execution")
    if not isinstance(technical, Mapping) or technical.get(
        "output_token_fallback_used"
    ) is not True:
        return
    if not path.is_file():
        raise LLMProtocolError(f"{context} lacks its fallback reservation journal")
    document = load_json(path)
    reservations = document.get("reservations")
    expected_count = technical.get("cumulative_output_token_fallback_attempts")
    expected = {
        "schema_version": FALLBACK_RESERVATION_SCHEMA_VERSION,
        "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
        "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "method": record.get("method"),
        "evaluation": record.get("evaluation"),
        "fold": record.get("fold"),
        "search_rank": record.get("search_rank"),
        "base_request_sha256": record.get("request_sha256"),
        "reserved_fallback_attempts": expected_count,
    }
    mismatches = {
        field: {"expected": value, "observed": document.get(field)}
        for field, value in expected.items()
        if document.get(field) != value
    }
    if isinstance(expected_count, int) and expected_count > MAX_FALLBACK_ATTEMPTS_PER_CASE:
        for field, value in {
            "technical_exception_id": RANK_1340_EXCEPTION_ID,
            "technical_exception_sha256": (
                EXPECTED_RANK_1340_EXCEPTION_ADDENDUM_SHA256
            ),
        }.items():
            if document.get(field) != value:
                mismatches[field] = {
                    "expected": value,
                    "observed": document.get(field),
                }
    if not isinstance(reservations, list) or len(reservations) != expected_count:
        mismatches["reservations"] = {
            "expected_count": expected_count,
            "observed": (
                len(reservations) if isinstance(reservations, list) else None
            ),
        }
    elif not reservations or reservations[-1].get(
        "actual_request_sha256"
    ) != technical.get("actual_request_sha256"):
        mismatches["actual_request_sha256"] = {
            "expected": technical.get("actual_request_sha256"),
            "observed": (
                reservations[-1].get("actual_request_sha256")
                if reservations
                else None
            ),
        }
    if mismatches:
        raise LLMProtocolError(
            f"{context} fallback journal mismatch: " + canonical_json(mismatches)
        )


def validate_success_primary_recovery_journal(
    record: Mapping[str, Any], *, path: Path, context: str
) -> None:
    if not (
        record.get("runner_version") == VERSION
        and record.get("method") == "M3"
        and record.get("evaluation") == "A1"
        and record.get("fold") is None
        and record.get("search_rank") == 551
    ):
        return
    if not path.is_file():
        raise LLMProtocolError(f"{context} lacks rank 551's 512 reservation journal")
    document = load_json(path)
    expected = {
        "schema_version": PRIMARY_RECOVERY_RESERVATION_SCHEMA_VERSION,
        "technical_amendment_id": TECHNICAL_AMENDMENT_ID,
        "technical_amendment_sha256": EXPECTED_TECHNICAL_AMENDMENT_SHA256,
        "status": "PRIMARY_RECOVERY_CALL_DURABLY_RESERVED",
        "method": "M3",
        "evaluation": "A1",
        "fold": None,
        "search_rank": 551,
        "base_request_sha256": record.get("request_sha256"),
        "actual_request_sha256": record.get("request_sha256"),
        "max_output_tokens": INITIAL_MAX_OUTPUT_TOKENS,
    }
    mismatches = {
        field: {"expected": value, "observed": document.get(field)}
        for field, value in expected.items()
        if document.get(field) != value
    }
    if mismatches:
        raise LLMProtocolError(
            f"{context} primary-recovery journal mismatch: "
            + canonical_json(mismatches)
        )


def _load_existing_success(
    path: Path,
    *,
    case: Mapping[str, Any],
    request: Mapping[str, Any],
    spec: RunSpec,
    fallback_reservation_path: Path | None = None,
    primary_recovery_reservation_path: Path | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = load_json(path)
    expected = {
        "status": "SUCCESS_VALIDATED",
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "search_rank": int(case["search_rank"]),
        "request_sha256": request["request_sha256"],
        "builder_payload_sha256": request["builder_payload_sha256"],
        "builder_metadata_sha256": request["builder_metadata_sha256"],
        "builder_metadata": request["builder_metadata"],
    }
    mismatches = {
        key: {"expected": value, "observed": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise LLMProtocolError(
            f"Existing success state conflicts for rank {case['search_rank']}: "
            f"{canonical_json(mismatches)}"
        )
    validate_persisted_success_record(
        record,
        request=request,
        context=f"Existing success state rank {case['search_rank']}",
    )
    technical = record.get("technical_execution")
    if isinstance(technical, Mapping) and technical.get(
        "output_token_fallback_used"
    ) is True:
        if fallback_reservation_path is None:
            raise LLMProtocolError(
                f"Existing amended fallback success rank {case['search_rank']} "
                "was loaded without its reservation path"
            )
        count = _fallback_reservation_count(
            fallback_reservation_path,
            case=case,
            request=request,
            spec=spec,
        )
        if count != technical.get("cumulative_output_token_fallback_attempts"):
            raise LLMProtocolError(
                f"Existing fallback reservation count differs for rank "
                f"{case['search_rank']}"
            )
    if (
        record.get("runner_version") == VERSION
        and spec.method == "M3"
        and spec.evaluation == "A1"
        and spec.fold is None
        and int(case["search_rank"]) == 551
    ):
        if primary_recovery_reservation_path is None:
            raise LLMProtocolError(
                "Existing amended rank 551 success was loaded without its "
                "primary reservation path"
            )
        validate_success_primary_recovery_journal(
            record,
            path=primary_recovery_reservation_path,
            context="Existing success state rank 551",
        )
    metadata = record.get("builder_metadata")
    if not isinstance(metadata, Mapping) or record.get(
        "builder_metadata_sha256"
    ) != sha256_text(canonical_json(metadata)):
        raise LLMProtocolError(
            f"Existing success state has invalid builder provenance for rank "
            f"{case['search_rank']}"
        )
    return record


def _record_failure_history(
    path: Path, failure: Mapping[str, Any], *, secret: str
) -> None:
    history: list[dict[str, Any]] = []
    if path.is_file():
        existing = load_json(path)
        value = existing.get("attempt_history", [])
        if isinstance(value, list):
            history = [item for item in value if isinstance(item, dict)]
    history.append(dict(failure))
    atomic_json(
        path,
        {
            "status": "UNRESOLVED_FAILURE_HISTORY",
            "search_rank": failure["search_rank"],
            "attempt_history": history,
            "latest_failure": failure,
        },
        secret=secret,
    )


def materialize_state(
    spec: RunSpec,
    cases: Sequence[Mapping[str, Any]],
    *,
    secret: str | None,
    started_at: str,
    interrupted: bool = False,
) -> dict[str, Any]:
    expected_ranks = {int(case["search_rank"]) for case in cases}
    success_dir = spec.state_dir / "success"
    failure_dir = spec.state_dir / "failures"
    successes: list[dict[str, Any]] = []
    unresolved_failures: list[dict[str, Any]] = []
    failure_history_ranks: list[int] = []
    for rank in sorted(expected_ranks):
        success_path = _state_path(success_dir, rank)
        failure_path = _state_path(failure_dir, rank)
        if success_path.is_file():
            successes.append(load_json(success_path))
        elif failure_path.is_file():
            failure = load_json(failure_path)
            latest = failure.get("latest_failure")
            if isinstance(latest, dict):
                unresolved_failures.append(latest)
        if failure_path.is_file():
            failure_history_ranks.append(rank)
    successes.sort(key=lambda row: int(row["search_rank"]))
    unresolved_failures.sort(key=lambda row: int(row["search_rank"]))
    if successes:
        atomic_jsonl(spec.output_path, successes, secret=secret)
    atomic_jsonl(spec.failure_manifest_path, unresolved_failures, secret=secret)
    completed = {int(row["search_rank"]) for row in successes}
    failed = {int(row["search_rank"]) for row in unresolved_failures}
    missing = sorted(expected_ranks - completed - failed)
    if interrupted:
        status = "INTERRUPTED_RESUMABLE"
    elif len(completed) == len(expected_ranks):
        status = "COMPLETE"
    elif failed:
        status = "COMPLETE_WITH_UNRESOLVED_FAILURES"
    else:
        status = "INCOMPLETE"
    diagnostics = {
        "execution_schema_version": EXECUTION_SCHEMA_VERSION,
        "runner_version": VERSION,
        "status": status,
        "setting_id": spec.setting_id,
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "dry_run": spec.dry_run,
        "started_at": started_at,
        "updated_at": utc_now(),
        "expected_cases": len(expected_ranks),
        "successful_cases": len(completed),
        "unresolved_failure_cases": len(failed),
        "missing_unattempted_cases": len(missing),
        "successful_search_ranks": sorted(completed),
        "unresolved_failure_search_ranks": sorted(failed),
        "missing_unattempted_search_ranks": missing,
        "failure_history_search_ranks": failure_history_ranks,
        "canonical_prediction_path": str(spec.output_path),
        "canonical_prediction_sha256": (
            sha256_file(spec.output_path) if spec.output_path.is_file() else None
        ),
        "canonical_prediction_rows": len(successes),
        "failure_manifest_path": str(spec.failure_manifest_path),
        "resume_rule": "SKIP_VALIDATED_SUCCESS; RETRY_FAILED_OR_MISSING",
    }
    atomic_json(spec.diagnostics_path, diagnostics, secret=secret)
    return diagnostics


def _m2_split_membership_sha256(
    evaluation: str, fold: int | None
) -> tuple[str, str, list[dict[str, str]]]:
    split_path = DEFAULT_A1_SPLIT if evaluation == "A1" else DEFAULT_A2_SPLIT
    rows = load_csv(split_path)
    if evaluation == "A2":
        rows = [row for row in rows if int(row.get("fold_id") or 0) == fold]
    rows = sorted(rows, key=lambda row: int(row["search_rank"]))
    if len(rows) != EXPECTED_BENCHMARK_N:
        raise LLMProtocolError(
            f"M2 prerequisite split {evaluation} fold {fold} has {len(rows)} rows"
        )
    payload = "".join(
        "\t".join(
            map(
                str,
                (
                    int(row["search_rank"]),
                    row["canonical_url"],
                    _split_role(row),
                    row.get("effective_supervised_train", ""),
                    row.get("fold_id", ""),
                ),
            )
        )
        + "\n"
        for row in rows
    )
    return sha256_text(payload), sha256_file(split_path), rows


def _validate_ordered_amp_labels(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise LLMProtocolError(f"{context} is not an AMP-label array")
    if len(value) != len(set(value)) or set(value) - set(AMP_LABEL_IDS):
        raise LLMProtocolError(f"{context} contains duplicate or unknown labels")
    expected = [label for label in AMP_LABEL_IDS if label in set(value)]
    if value != expected:
        raise LLMProtocolError(f"{context} is not in frozen ontology order")
    return value


def validate_m2_completion_gate(
    *,
    model_root: Path = DEFAULT_M2_MODEL_ROOT,
    prediction_root: Path = DEFAULT_M2_PREDICTION_ROOT,
) -> dict[str, Any]:
    """Require complete, internally consistent M2 A1 and all three A2 runs."""

    require_file_sha256(DEFAULT_M2_CONFIG, EXPECTED_M2_CONFIG_SHA256, "M2 config")
    benchmark = load_benchmark_index()
    settings = (("A1", None), ("A2", 1), ("A2", 2), ("A2", 3))
    provenance: list[dict[str, Any]] = []
    for evaluation, fold in settings:
        key = "a1" if evaluation == "A1" else f"a2_fold_{fold}"
        metadata_path = model_root / key / "run_metadata.json"
        prediction_path = prediction_root / f"{key}_test_predictions.jsonl"
        if not metadata_path.is_file():
            raise LLMProtocolError(
                f"Paid LLM execution is blocked until M2 {key} metadata exists: "
                f"{metadata_path}"
            )
        metadata = load_json(metadata_path)
        membership_sha, split_sha, _split_rows = _m2_split_membership_sha256(
            evaluation, fold
        )
        setting_rows = load_setting_rows(evaluation, fold, benchmark)
        expected_cases = {
            int(case["search_rank"]): case
            for case in setting_rows
            if case["role"] == "TEST"
        }
        expected_metadata = {
            "artifact_schema_version": "sherloc-m2-artifacts-v1",
            "method_id": "M2",
            "evaluation": evaluation,
            "fold": fold,
            "primary_cohort_id": EXPECTED_COHORT_ID,
            "label_order": list(AMP_LABEL_IDS),
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "ontology_sha256": EXPECTED_ONTOLOGY_SHA256,
            "config_sha256": EXPECTED_M2_CONFIG_SHA256,
            "split_file_sha256": split_sha,
            "split_membership_sha256": membership_sha,
            "test_n": len(expected_cases),
            "status": "COMPLETE",
            "prediction_rows": len(expected_cases),
            "test_labels_used_for_selection": False,
        }
        mismatches = {
            field: {"expected": expected, "observed": metadata.get(field)}
            for field, expected in expected_metadata.items()
            if metadata.get(field) != expected
        }
        if mismatches:
            raise LLMProtocolError(
                f"Paid LLM execution is blocked by incomplete/invalid M2 {key}: "
                + canonical_json(mismatches)
            )
        if not prediction_path.is_file():
            raise LLMProtocolError(f"M2 {key} prediction file is missing: {prediction_path}")
        prediction_sha = sha256_file(prediction_path)
        if metadata.get("prediction_sha256") != prediction_sha:
            raise LLMProtocolError(f"M2 {key} prediction SHA-256 differs from metadata")
        recorded_prediction = Path(str(metadata.get("prediction_path") or ""))
        if not recorded_prediction.is_absolute():
            recorded_prediction = REPO_ROOT / recorded_prediction
        if recorded_prediction.resolve() != prediction_path.resolve():
            raise LLMProtocolError(f"M2 {key} metadata names another prediction path")
        rows = load_jsonl(prediction_path)
        observed: dict[int, dict[str, Any]] = {}
        for row in rows:
            rank = int(row.get("search_rank") or 0)
            if rank in observed:
                raise LLMProtocolError(f"M2 {key} duplicates search rank {rank}")
            observed[rank] = row
        if set(observed) != set(expected_cases):
            raise LLMProtocolError(f"M2 {key} TEST membership differs from final split")
        for rank, row in observed.items():
            case = expected_cases[rank]
            expected_row = {
                "method_id": "M2",
                "evaluation": evaluation,
                "fold": fold,
                "split": "TEST",
                "canonical_url": case["canonical_url"],
                "jurisdiction": case["jurisdiction"],
                "fact_summary": case["fact_summary"],
                "input_sha256": sha256_text(case["fact_summary"]),
                "silver_reference_labels": case["silver_reference_labels"],
                "primary_cohort_id": EXPECTED_COHORT_ID,
                "config_sha256": EXPECTED_M2_CONFIG_SHA256,
                "split_membership_sha256": membership_sha,
            }
            bad = [field for field, value in expected_row.items() if row.get(field) != value]
            if bad:
                raise LLMProtocolError(
                    f"M2 {key} prediction provenance differs at rank {rank}: {bad}"
                )
            _validate_ordered_amp_labels(
                row.get("predicted_labels"), context=f"M2 {key} rank {rank} prediction"
            )
            probabilities = row.get("probabilities_by_label")
            if not isinstance(probabilities, Mapping) or set(probabilities) != set(
                AMP_LABEL_IDS
            ):
                raise LLMProtocolError(f"M2 {key} rank {rank} has invalid probabilities")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
                for value in probabilities.values()
            ):
                raise LLMProtocolError(
                    f"M2 {key} rank {rank} has non-finite/out-of-range probabilities"
                )
        provenance.append(
            {
                "setting": key,
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": prediction_sha,
                "prediction_rows": len(rows),
            }
        )
    return {"status": "M2_A1_A2_COMPLETE", "settings": provenance}


def validate_completed_llm_setting(
    spec: RunSpec,
    *,
    benchmark: Mapping[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a complete canonical M3/M4 setting before a dependent spend."""

    if spec.dry_run:
        raise LLMProtocolError("Dry-run output cannot satisfy a primary stage gate")
    require_file_sha256(DEFAULT_CONFIG, EXPECTED_CONFIG_SHA256, "LLM config")
    benchmark_index = dict(benchmark or load_benchmark_index())
    expected_cases = {
        int(case["search_rank"]): case
        for case in load_setting_rows(spec.evaluation, spec.fold, benchmark_index)
        if case["role"] == "TEST"
    }
    if not spec.diagnostics_path.is_file():
        raise LLMProtocolError(
            f"Required prior setting {spec.setting_id} lacks diagnostics: "
            f"{spec.diagnostics_path}"
        )
    diagnostics = load_json(spec.diagnostics_path)
    expected_diagnostics = {
        "status": "COMPLETE",
        "setting_id": spec.setting_id,
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "dry_run": False,
        "expected_cases": len(expected_cases),
        "successful_cases": len(expected_cases),
        "unresolved_failure_cases": 0,
        "missing_unattempted_cases": 0,
        "canonical_prediction_path": str(spec.output_path),
        "canonical_prediction_rows": len(expected_cases),
    }
    mismatches = {
        field: {"expected": expected, "observed": diagnostics.get(field)}
        for field, expected in expected_diagnostics.items()
        if diagnostics.get(field) != expected
    }
    if mismatches:
        raise LLMProtocolError(
            f"Required prior setting {spec.setting_id} is not complete: "
            + canonical_json(mismatches)
        )
    lock_metadata = diagnostics.get("run_lock")
    if not isinstance(lock_metadata, Mapping):
        raise LLMProtocolError(f"{spec.setting_id} lacks run-lock provenance")
    lock_history_path = Path(str(lock_metadata.get("history_path") or ""))
    if not lock_history_path.is_file():
        raise LLMProtocolError(f"{spec.setting_id} run-lock history is missing")
    lock_history = load_json(lock_history_path)
    if (
        lock_history.get("status") != "RELEASED"
        or lock_history.get("token") != lock_metadata.get("token")
        or lock_history.get("setting_id") != spec.setting_id
    ):
        raise LLMProtocolError(f"{spec.setting_id} run-lock provenance is invalid")
    stage_provenance = diagnostics.get("stage_prerequisite_provenance")
    m2_provenance = (
        stage_provenance.get("m2") if isinstance(stage_provenance, Mapping) else None
    )
    if not isinstance(m2_provenance, Mapping) or m2_provenance.get(
        "status"
    ) != "M2_A1_A2_COMPLETE":
        raise LLMProtocolError(f"{spec.setting_id} lacks the mandatory M2 stage proof")
    if spec.method == "M4" and spec.evaluation == "A1":
        m3_a1 = stage_provenance.get("m3_a1")
        if not isinstance(m3_a1, Mapping) or m3_a1.get("status") != "COMPLETE":
            raise LLMProtocolError(f"{spec.setting_id} lacks its M3 A1 stage proof")
    if spec.evaluation == "A2":
        a1_metrics = stage_provenance.get("a1_metrics")
        if not isinstance(a1_metrics, Mapping) or a1_metrics.get("status") != (
            "CANONICAL_A1_M1_M4_METRICS_COMPLETE"
        ):
            raise LLMProtocolError(f"{spec.setting_id} lacks its canonical A1 metric proof")
        a1_settings = stage_provenance.get("a1_llm_settings")
        if not isinstance(a1_settings, list) or len(a1_settings) != 2 or any(
            not isinstance(item, Mapping) or item.get("status") != "COMPLETE"
            for item in a1_settings
        ):
            raise LLMProtocolError(f"{spec.setting_id} lacks complete A1 LLM stage proof")
    if spec.method == "M4" and spec.evaluation == "A2":
        m3_a2 = stage_provenance.get("m3_a2")
        if not isinstance(m3_a2, list) or len(m3_a2) != 3 or any(
            not isinstance(item, Mapping) or item.get("status") != "COMPLETE"
            for item in m3_a2
        ):
            raise LLMProtocolError(f"{spec.setting_id} lacks all three M3 A2 proofs")
    if not spec.output_path.is_file():
        raise LLMProtocolError(f"Required prediction file is missing: {spec.output_path}")
    prediction_sha = sha256_file(spec.output_path)
    if diagnostics.get("canonical_prediction_sha256") != prediction_sha:
        raise LLMProtocolError(
            f"Required prior setting {spec.setting_id} prediction hash changed"
        )
    rows = load_jsonl(spec.output_path)
    observed: dict[int, dict[str, Any]] = {}
    for row in rows:
        rank = int(row.get("search_rank") or 0)
        if rank in observed:
            raise LLMProtocolError(f"{spec.setting_id} duplicates rank {rank}")
        observed[rank] = row
    if set(observed) != set(expected_cases):
        raise LLMProtocolError(f"{spec.setting_id} TEST membership is not canonical")
    config = load_json(DEFAULT_CONFIG)
    for rank, row in observed.items():
        case = expected_cases[rank]
        expected_row = {
            "status": "SUCCESS_VALIDATED",
            "method": spec.method,
            "method_id": spec.method,
            "evaluation": spec.evaluation,
            "fold": spec.fold,
            "split": "TEST",
            "canonical_url": case["canonical_url"],
            "jurisdiction": case["jurisdiction"],
            "fact_summary": case["fact_summary"],
            "input_sha256": sha256_text(case["fact_summary"]),
            "silver_reference_labels": case["silver_reference_labels"],
            "primary_cohort_id": EXPECTED_COHORT_ID,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "prompt_sha256": config["methods"][spec.method]["prompt_sha256"],
            "schema_sha256": config["structured_output"]["schema_sha256"],
            "split_membership_sha256": target_membership_hash(
                list(expected_cases.values()), spec
            ),
        }
        bad = [field for field, value in expected_row.items() if row.get(field) != value]
        if bad:
            raise LLMProtocolError(
                f"{spec.setting_id} prediction provenance differs at rank {rank}: {bad}"
            )
        validate_persisted_success_record(
            row,
            request=None,
            context=f"{spec.setting_id} rank {rank}",
        )
        if row.get("request_sha256") != row.get("request_payload_sha256"):
            raise LLMProtocolError(f"{spec.setting_id} request hash mismatch at rank {rank}")
        validate_success_fallback_journal(
            row,
            path=_state_path(spec.state_dir / "fallback_reservations", rank),
            context=f"{spec.setting_id} rank {rank}",
        )
        validate_success_primary_recovery_journal(
            row,
            path=_state_path(
                spec.state_dir / "primary_recovery_reservations", rank
            ),
            context=f"{spec.setting_id} rank {rank}",
        )
        state_path = _state_path(spec.state_dir / "success", rank)
        if not state_path.is_file() or load_json(state_path) != row:
            raise LLMProtocolError(
                f"{spec.setting_id} canonical/state trace differs at rank {rank}"
            )
    return {
        "status": "COMPLETE",
        "setting_id": spec.setting_id,
        "prediction_path": str(spec.output_path),
        "prediction_sha256": prediction_sha,
        "prediction_rows": len(rows),
        "diagnostics_path": str(spec.diagnostics_path),
        "diagnostics_sha256": sha256_file(spec.diagnostics_path),
    }


def validate_canonical_a1_metrics_gate(
    *,
    metric_root: Path = DEFAULT_METRIC_ROOT,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
) -> dict[str, Any]:
    """Require canonical complete M1--M4 A1 metrics before any A2 request."""

    if metric_root.resolve() != DEFAULT_METRIC_ROOT.resolve():
        raise LLMProtocolError("A2 requires the canonical default A1 metrics directory")
    manifest_path = metric_root / "amp_evaluation_manifest.json"
    if not manifest_path.is_file():
        raise LLMProtocolError(
            f"A2 execution is blocked until canonical A1 metrics exist: {manifest_path}"
        )
    manifest = load_json(manifest_path)
    a1 = manifest.get("evaluations", {}).get("A1", {})
    split = manifest.get("split_validation", {})
    required_methods = {"M1", "M2", "M3", "M4"}
    if (
        set(a1.get("methods", [])) != required_methods
        or a1.get("test_n") != 253
        or a1.get("macro_label_count") != 17
        or a1.get("macro_label_ids") != list(AMP_LABEL_IDS)
        or split.get("a1_final_split_validated") is not True
        or split.get("a1_expected_test_n") != 253
    ):
        raise LLMProtocolError("Canonical A1 metric manifest is incomplete or stale")
    expected_inputs = {
        (prediction_root / method.lower() / "a1_test_predictions.jsonl").resolve()
        for method in required_methods
    }
    input_entries = manifest.get("input_files")
    if not isinstance(input_entries, list):
        raise LLMProtocolError("Canonical A1 metric manifest lacks input-file hashes")
    observed_inputs: dict[Path, str] = {}
    for raw in input_entries:
        if not isinstance(raw, Mapping):
            raise LLMProtocolError("Malformed metric input-file provenance")
        path = Path(str(raw.get("path") or ""))
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved = path.resolve()
        if resolved in observed_inputs:
            raise LLMProtocolError(f"Metric manifest duplicates input path {resolved}")
        observed_inputs[resolved] = str(raw.get("sha256") or "")
    for path in expected_inputs:
        if path not in observed_inputs or not path.is_file():
            raise LLMProtocolError(f"Canonical A1 metric input is absent: {path}")
        if observed_inputs[path] != sha256_file(path):
            raise LLMProtocolError(f"Canonical A1 metric input hash changed: {path}")
    metric_files = {
        "primary": metric_root / "a1/amp_primary_results.csv",
        "per_label": metric_root / "a1/amp_per_label.csv",
        "bootstrap": metric_root / "a1/amp_bootstrap_cis.csv",
        "case_errors": metric_root / "a1/amp_case_level_errors.csv",
    }
    for name, path in metric_files.items():
        if not path.is_file():
            raise LLMProtocolError(f"Canonical A1 {name} metrics are missing: {path}")
    primary = load_csv(metric_files["primary"])
    primary_rows = [row for row in primary if row.get("prediction_variant") == "PRIMARY"]
    try:
        primary_label_orders = [
            json.loads(row.get("macro_label_ids_json") or "null")
            for row in primary_rows
        ]
    except json.JSONDecodeError as exc:
        raise LLMProtocolError("Canonical A1 primary labels are malformed JSON") from exc
    if (
        len(primary_rows) != 4
        or {row.get("method") for row in primary_rows} != required_methods
        or any(row.get("test_n") != "253" for row in primary_rows)
        or any(row.get("macro_label_count") != "17" for row in primary_rows)
        or any(labels != list(AMP_LABEL_IDS) for labels in primary_label_orders)
    ):
        raise LLMProtocolError("Canonical A1 primary metric table is incomplete")
    per_label = load_csv(metric_files["per_label"])
    expected_method_labels = {
        (method, label) for method in required_methods for label in AMP_LABEL_IDS
    }
    if (
        len(per_label) != 4 * len(AMP_LABEL_IDS)
        or {(row.get("method"), row.get("label_id")) for row in per_label}
        != expected_method_labels
    ):
        raise LLMProtocolError("Canonical A1 per_label metric table is incomplete")
    bootstrap = load_csv(metric_files["bootstrap"])
    expected_bootstrap = {
        (method, metric)
        for method in required_methods
        for metric in ("macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard")
    }
    if (
        len(bootstrap) != 16
        or {(row.get("method"), row.get("metric")) for row in bootstrap}
        != expected_bootstrap
    ):
        raise LLMProtocolError("Canonical A1 bootstrap metric table is incomplete")
    case_errors = load_csv(metric_files["case_errors"])
    if (
        len(case_errors) != 4 * 253
        or {row.get("method") for row in case_errors} != required_methods
        or any(row.get("split") != "TEST" for row in case_errors)
    ):
        raise LLMProtocolError("Canonical A1 case_errors metric table is incomplete")
    return {
        "status": "CANONICAL_A1_M1_M4_METRICS_COMPLETE",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metric_files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in metric_files.items()
        },
    }


def validate_stage_prerequisites(
    spec: RunSpec,
    *,
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    log_root: Path = DEFAULT_LOG_ROOT,
    metric_root: Path = DEFAULT_METRIC_ROOT,
    m2_model_root: Path = DEFAULT_M2_MODEL_ROOT,
    m2_prediction_root: Path = DEFAULT_M2_PREDICTION_ROOT,
) -> dict[str, Any]:
    """Enforce the frozen paid-execution sequence immediately before requests."""

    provenance: dict[str, Any] = {
        "m2": validate_m2_completion_gate(
            model_root=m2_model_root, prediction_root=m2_prediction_root
        )
    }
    if spec.dry_run:
        return provenance
    if spec.evaluation == "A1" and spec.method == "M4":
        prior = make_spec(
            "M3",
            "A1",
            None,
            dry_run=False,
            prediction_root=prediction_root,
            log_root=log_root,
        )
        provenance["m3_a1"] = validate_completed_llm_setting(prior)
    if spec.evaluation == "A2":
        provenance["a1_llm_settings"] = [
            validate_completed_llm_setting(
                make_spec(
                    method,
                    "A1",
                    None,
                    dry_run=False,
                    prediction_root=prediction_root,
                    log_root=log_root,
                )
            )
            for method in ("M3", "M4")
        ]
        provenance["a1_metrics"] = validate_canonical_a1_metrics_gate(
            metric_root=metric_root, prediction_root=prediction_root
        )
        if spec.method == "M4":
            provenance["m3_a2"] = [
                validate_completed_llm_setting(
                    make_spec(
                        "M3",
                        "A2",
                        fold,
                        dry_run=False,
                        prediction_root=prediction_root,
                        log_root=log_root,
                    )
                )
                for fold in (1, 2, 3)
            ]
    return provenance


def validate_prepared_run_inputs(
    spec: RunSpec,
    cases: Sequence[Mapping[str, Any]],
    *,
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
    heldout_jurisdictions: Sequence[str],
    config: Mapping[str, Any],
    demo_bank: Mapping[str, Any],
) -> None:
    """Re-derive all mutable prepared objects from canonical frozen artifacts."""

    benchmark = load_benchmark_index()
    setting_rows = load_setting_rows(
        "A1" if spec.dry_run else spec.evaluation,
        None if spec.dry_run else spec.fold,
        benchmark,
    )
    if spec.dry_run:
        approved = [int(item["search_rank"]) for item in demo_bank["approved_cases"]]
        expected_cases = deterministic_dry_run_cases(
            setting_rows,
            count=len(cases),
            excluded_ranks=approved,
        )
        expected_heldout: list[str] = []
    else:
        expected_cases = [dict(case) for case in setting_rows if case["role"] == "TEST"]
        expected_heldout = (
            sorted({case["jurisdiction"] for case in expected_cases})
            if spec.evaluation == "A2"
            else []
        )
    if canonical_json(list(cases)) != canonical_json(expected_cases):
        raise LLMProtocolError(
            f"Prepared case objects/membership changed for {spec.setting_id}"
        )
    if list(heldout_jurisdictions) != expected_heldout:
        raise LLMProtocolError(
            f"Prepared held-out jurisdictions changed for {spec.setting_id}"
        )
    if spec.method == "M3":
        if demos is not None or demo_metadata is not None:
            raise LLMProtocolError("M3 unexpectedly received demonstration objects")
        return
    if spec.bank_id is None:
        raise LLMProtocolError("M4 setting lacks a canonical demo-bank ID")
    expected_demos, expected_metadata = load_demo_bank_for_setting(
        spec.bank_id,
        demo_bank,
        config,
        benchmark,
        actual_test_jurisdictions=expected_heldout,
    )
    if canonical_json(demos) != canonical_json(expected_demos):
        raise LLMProtocolError(f"Prepared M4 demos changed for {spec.setting_id}")
    if canonical_json(demo_metadata) != canonical_json(expected_metadata):
        raise LLMProtocolError(f"Prepared M4 demo metadata changed for {spec.setting_id}")


def _execute_cases_under_lock(
    spec: RunSpec,
    cases: Sequence[dict[str, Any]],
    *,
    client: Any,
    secret: str,
    sdk_version: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
    heldout_jurisdictions: Sequence[str],
    config_path: Path,
    m3_prompt_path: Path,
    m4_prompt_path: Path,
    max_attempts: int,
    base_backoff_seconds: float,
    workers: int = DEFAULT_WORKERS,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise LLMProtocolError(f"workers must be between 1 and {MAX_WORKERS}")
    previous_prerequisite_provenance: Mapping[str, Any] | None = None
    if spec.diagnostics_path.is_file():
        previous_diagnostics = load_json(spec.diagnostics_path)
        previous = previous_diagnostics.get("stage_prerequisite_provenance")
        if isinstance(previous, Mapping):
            previous_prerequisite_provenance = previous
    started_at = utc_now()
    success_dir = spec.state_dir / "success"
    failure_dir = spec.state_dir / "failures"
    fallback_reservation_dir = spec.state_dir / "fallback_reservations"
    primary_recovery_reservation_dir = (
        spec.state_dir / "primary_recovery_reservations"
    )
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    fallback_reservation_dir.mkdir(parents=True, exist_ok=True)
    primary_recovery_reservation_dir.mkdir(parents=True, exist_ok=True)
    membership_sha256 = target_membership_hash(cases, spec)
    interrupted = False
    fatal = False
    pending: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ] = []
    prerequisite_provenance: Mapping[str, Any] | None = None
    try:
        # Request construction and resume-state validation stay serialized.
        # This catches drift before any new request is sent.
        for case in sorted(cases, key=lambda row: int(row["search_rank"])):
            request = build_request_for_case(
                spec,
                case,
                demos=demos,
                demo_metadata=demo_metadata,
                heldout_jurisdictions=heldout_jurisdictions,
                effective_model_id=str(model_marker["effective_model_id"]),
                contract=contract,
                config=config,
                config_path=config_path,
                m3_prompt_path=m3_prompt_path,
                m4_prompt_path=m4_prompt_path,
            )
            success_path = _state_path(success_dir, int(case["search_rank"]))
            existing = _load_existing_success(
                success_path,
                case=case,
                request=request,
                spec=spec,
                fallback_reservation_path=_state_path(
                    fallback_reservation_dir, int(case["search_rank"])
                ),
                primary_recovery_reservation_path=_state_path(
                    primary_recovery_reservation_dir,
                    int(case["search_rank"]),
                ),
            )
            if existing is not None:
                continue
            resume_policy = resolve_failure_resume_policy(
                _state_path(failure_dir, int(case["search_rank"])),
                case=case,
                request=request,
                spec=spec,
                fallback_reservation_path=_state_path(
                    fallback_reservation_dir, int(case["search_rank"])
                ),
                primary_recovery_reservation_path=_state_path(
                    primary_recovery_reservation_dir,
                    int(case["search_rank"]),
                ),
            )
            pending.append((case, request, resume_policy))

        validate_canonical_m3_a1_amendment_scope(spec, pending)

        # Close the build-time check/use window.  Revalidate every canonical
        # byte source, reload the builder contract, then rebuild each pending
        # payload and compare both its payload and metadata hashes.  No API call
        # can occur before this block completes.
        validate_canonical_artifact_hashes()
        fresh_contract, fresh_config, fresh_bank = validate_frozen_contract(
            config_path=config_path,
            demo_bank_path=DEFAULT_DEMO_BANK,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
            ontology_path=DEFAULT_ONTOLOGY,
            review_path=DEFAULT_REVIEW,
        )
        if (
            fresh_contract["config_sha256"] != contract["config_sha256"]
            or fresh_contract["marked_block_sha256"]
            != contract["marked_block_sha256"]
            or canonical_json(fresh_config) != canonical_json(config)
        ):
            raise LLMProtocolError("Frozen request contract changed during run preparation")
        validate_prepared_run_inputs(
            spec,
            cases,
            demos=demos,
            demo_metadata=demo_metadata,
            heldout_jurisdictions=heldout_jurisdictions,
            config=fresh_config,
            demo_bank=fresh_bank,
        )
        for case, request, _resume_policy in pending:
            revalidate_built_request(
                request,
                spec,
                case,
                demos=demos,
                demo_metadata=demo_metadata,
                heldout_jurisdictions=heldout_jurisdictions,
                effective_model_id=str(model_marker["effective_model_id"]),
                contract=fresh_contract,
                config=fresh_config,
                config_path=config_path,
                m3_prompt_path=m3_prompt_path,
                m4_prompt_path=m4_prompt_path,
            )
        validate_canonical_artifact_hashes()
        if pending:
            inferred_prediction_root = (
                DEFAULT_PREDICTION_ROOT
                if spec.dry_run
                else spec.output_path.parent.parent
            )
            inferred_log_root = (
                spec.diagnostics_path.parent.parent
                if spec.dry_run
                else spec.diagnostics_path.parent
            )
            prerequisite_provenance = validate_stage_prerequisites(
                spec,
                prediction_root=inferred_prediction_root,
                log_root=inferred_log_root,
                metric_root=DEFAULT_METRIC_ROOT,
                m2_model_root=DEFAULT_M2_MODEL_ROOT,
                m2_prediction_root=DEFAULT_M2_PREDICTION_ROOT,
            )
            # The stage checks read independent artifacts and may take several
            # seconds.  Re-pin canonical request sources once more immediately
            # before workers can issue requests.
            validate_canonical_artifact_hashes()

        def invoke_task(
            task: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            case, request, resume_policy = task

            def reserve_attempt(
                attempt_number: int, actual_payload: Mapping[str, Any]
            ) -> None:
                reserve_fallback_attempt(
                    _state_path(
                        fallback_reservation_dir, int(case["search_rank"])
                    ),
                    case=case,
                    request=request,
                    spec=spec,
                    fallback_attempt_number=attempt_number,
                    actual_payload=actual_payload,
                    secret=secret,
                    technical_exception_id=resume_policy[
                        "technical_exception_id"
                    ],
                    technical_exception_sha256=resume_policy[
                        "technical_exception_sha256"
                    ],
                )

            def reserve_primary_attempt(
                attempt_number: int, actual_payload: Mapping[str, Any]
            ) -> None:
                reserve_primary_recovery_attempt(
                    _state_path(
                        primary_recovery_reservation_dir,
                        int(case["search_rank"]),
                    ),
                    case=case,
                    request=request,
                    spec=spec,
                    primary_attempt_number=attempt_number,
                    actual_payload=actual_payload,
                    secret=secret,
                )

            result = invoke_with_retries(
                client,
                request["payload"],
                max_attempts=max_attempts,
                base_backoff_seconds=base_backoff_seconds,
                sleeper=sleeper,
                secret=secret,
                start_with_fallback=bool(
                    resume_policy["start_with_fallback"]
                ),
                prior_fallback_attempts=int(
                    resume_policy["prior_fallback_attempts"]
                ),
                prior_primary_incomplete_provenance=resume_policy[
                    "prior_primary_incomplete_provenance"
                ],
                primary_attempt_limit=resume_policy["primary_attempt_limit"],
                fallback_attempt_ceiling=int(
                    resume_policy["fallback_attempt_ceiling"]
                ),
                technical_exception_id=resume_policy["technical_exception_id"],
                technical_exception_sha256=resume_policy[
                    "technical_exception_sha256"
                ],
                fallback_attempt_reserver=reserve_attempt,
                primary_attempt_reserver=(
                    reserve_primary_attempt
                    if resume_policy["primary_recovery_required"]
                    else None
                ),
            )
            return case, request, result

        def commit_result(
            case: dict[str, Any], request: dict[str, Any], result: dict[str, Any]
        ) -> bool:
            """Commit on the main thread; return true for fatal access errors."""

            if result["ok"]:
                record = make_success_record(
                    spec,
                    case,
                    request,
                    result,
                    contract=contract,
                    config=config,
                    model_marker=model_marker,
                    sdk_version=sdk_version,
                    demo_metadata=demo_metadata,
                    split_membership_sha256=membership_sha256,
                )
                atomic_json(
                    _state_path(success_dir, int(case["search_rank"])),
                    record,
                    secret=secret,
                )
            else:
                failure = make_failure_record(
                    spec,
                    case,
                    request,
                    result,
                    contract=contract,
                    config=config,
                    model_marker=model_marker,
                    sdk_version=sdk_version,
                    demo_metadata=demo_metadata,
                )
                _record_failure_history(
                    _state_path(failure_dir, int(case["search_rank"])),
                    failure,
                    secret=secret,
                )
                if result.get("fatal_access_error"):
                    return True
            return False

        if workers == 1:
            for task in pending:
                case, request, result = invoke_task(task)
                if commit_result(case, request, result):
                    fatal = True
                    break
        else:
            # The official sync client uses a thread-safe HTTP transport.  A
            # worker may atomically reserve its own per-case 2048-call journal
            # immediately before that call; completed result commits remain on
            # the main thread.
            executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="sherloc-llm"
            )
            futures: dict[Future[Any], int] = {
                executor.submit(invoke_task, task): int(task[0]["search_rank"])
                for task in pending
            }
            processed: set[Future[Any]] = set()
            concurrent_interrupt = False
            concurrent_commit_errors: list[BaseException] = []
            try:
                for future in as_completed(futures):
                    case, request, result = future.result()
                    try:
                        result_is_fatal = commit_result(case, request, result)
                    except BaseException as exc:
                        concurrent_commit_errors.append(exc)
                        break
                    processed.add(future)
                    if result_is_fatal:
                        fatal = True
                        break
            except KeyboardInterrupt:
                concurrent_interrupt = True
            finally:
                # Stop queued work after a fatal access error or interrupt, but
                # wait for already-running requests.  Those requests may have
                # incurred cost and their validated results must be committed.
                if fatal or concurrent_interrupt or concurrent_commit_errors:
                    for future in futures:
                        if future not in processed:
                            future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                for future in futures:
                    if future in processed or future.cancelled():
                        continue
                    case, request, result = future.result()
                    try:
                        result_is_fatal = commit_result(case, request, result)
                    except BaseException as exc:
                        concurrent_commit_errors.append(exc)
                        # Continue draining: another in-flight request may have
                        # incurred cost and still needs a durable commit.
                        continue
                    processed.add(future)
                    if result_is_fatal:
                        fatal = True
            if concurrent_interrupt:
                raise KeyboardInterrupt
            if concurrent_commit_errors:
                raise concurrent_commit_errors[0]
    except KeyboardInterrupt:  # Preserve all completed per-case commits.
        interrupted = True
    diagnostics = materialize_state(
        spec,
        cases,
        secret=secret,
        started_at=started_at,
        interrupted=interrupted,
    )
    diagnostics = dict(diagnostics)
    diagnostics["worker_count"] = workers
    diagnostics["stage_prerequisite_provenance"] = (
        prerequisite_provenance or previous_prerequisite_provenance
    )
    if fatal:
        diagnostics["status"] = "BLOCKED_FATAL_API_ACCESS_ERROR"
    atomic_json(spec.diagnostics_path, diagnostics, secret=secret)
    return diagnostics


def execute_cases(
    spec: RunSpec,
    cases: Sequence[dict[str, Any]],
    *,
    client: Any,
    secret: str,
    sdk_version: str,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
    demos: Sequence[Mapping[str, Any]] | None,
    demo_metadata: Mapping[str, Any] | None,
    heldout_jurisdictions: Sequence[str],
    config_path: Path,
    m3_prompt_path: Path,
    m4_prompt_path: Path,
    max_attempts: int,
    base_backoff_seconds: float,
    workers: int = DEFAULT_WORKERS,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute one setting while holding its exclusive inter-process lock."""

    with SettingRunLock(spec) as run_lock:
        validate_canonical_artifact_hashes()
        diagnostics = _execute_cases_under_lock(
            spec,
            cases,
            client=client,
            secret=secret,
            sdk_version=sdk_version,
            contract=contract,
            config=config,
            model_marker=model_marker,
            demos=demos,
            demo_metadata=demo_metadata,
            heldout_jurisdictions=heldout_jurisdictions,
            config_path=config_path,
            m3_prompt_path=m3_prompt_path,
            m4_prompt_path=m4_prompt_path,
            max_attempts=max_attempts,
            base_backoff_seconds=base_backoff_seconds,
            workers=workers,
            sleeper=sleeper,
        )
        diagnostics = dict(diagnostics)
        diagnostics["run_lock"] = run_lock.diagnostic_metadata()
        atomic_json(spec.diagnostics_path, diagnostics, secret=secret)
        return diagnostics


def dry_run_gate_path(log_root: Path, method: str) -> Path:
    return log_root / f"{method.lower()}_dry_run_gate.json"


def write_dry_run_gate(
    path: Path,
    spec: RunSpec,
    diagnostics: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
    demo_metadata: Mapping[str, Any] | None,
    secret: str,
) -> dict[str, Any]:
    if diagnostics.get("status") != "COMPLETE":
        raise LLMProtocolError("Dry run did not pass; TEST execution remains blocked")
    if not (MIN_DRY_RUN_CASES <= len(cases) <= MAX_DRY_RUN_CASES):
        raise LLMProtocolError("Dry-run gate has an invalid case count")
    if any(case["role"] == "TEST" for case in cases):
        raise LLMProtocolError("Dry-run gate contains a TEST case")
    gate = {
        "schema_version": DRY_RUN_GATE_SCHEMA_VERSION,
        "runner_version": VERSION,
        "status": "DRY_RUN_GATE_PASSED",
        "passed_at": utc_now(),
        "method": spec.method,
        "case_count": len(cases),
        "search_ranks": [int(case["search_rank"]) for case in cases],
        "all_cases_non_test": True,
        "semantic_tuning_performed": False,
        "config_sha256": contract["config_sha256"],
        "prompt_sha256": config["methods"][spec.method]["prompt_sha256"],
        "schema_sha256": config["structured_output"]["schema_sha256"],
        "effective_model_id": model_marker["effective_model_id"],
        "model_access_marker_sha256": model_marker.get("marker_sha256"),
        "technical_checks": [
            "authentication",
            "model_access",
            "structured_output_schema",
            "ontology_ids",
            "response_parser",
            "atomic_logging",
            "request_response_mapping",
            "token_accounting",
        ],
        "demo_bank_id": (demo_metadata or {}).get("demo_bank_id"),
        "demo_bank_membership_sha256": (
            demo_metadata or {}
        ).get("demo_bank_membership_sha256"),
        "diagnostics_path": str(spec.diagnostics_path),
    }
    atomic_json(path, gate, secret=secret)
    return gate


def validate_dry_run_gate(
    path: Path,
    method: str,
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    model_marker: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise LLMProtocolError(
            f"{method} dry-run gate is absent; run --method {method} --dry-run first"
        )
    gate = load_json(path)
    expected = {
        "status": "DRY_RUN_GATE_PASSED",
        "method": method,
        "config_sha256": contract["config_sha256"],
        "prompt_sha256": config["methods"][method]["prompt_sha256"],
        "schema_sha256": config["structured_output"]["schema_sha256"],
        "effective_model_id": model_marker["effective_model_id"],
        "all_cases_non_test": True,
        "semantic_tuning_performed": False,
    }
    mismatches = {
        key: {"expected": value, "observed": gate.get(key)}
        for key, value in expected.items()
        if gate.get(key) != value
    }
    if mismatches:
        raise LLMProtocolError(
            f"{method} dry-run gate is stale or invalid: {canonical_json(mismatches)}"
        )
    count = int(gate.get("case_count") or 0)
    if count < MIN_DRY_RUN_CASES or count > MAX_DRY_RUN_CASES:
        raise LLMProtocolError("Dry-run gate case count is outside 3--5")
    return gate


def make_spec(
    method: str,
    evaluation: str,
    fold: int | None,
    *,
    dry_run: bool,
    prediction_root: Path,
    log_root: Path,
) -> RunSpec:
    normalized_method = method.upper()
    if normalized_method not in {"M3", "M4"}:
        raise LLMProtocolError("method must be M3 or M4")
    if dry_run:
        setting = f"dry_run_{normalized_method.lower()}"
        return RunSpec(
            method=normalized_method,
            evaluation="A1",
            fold=None,
            dry_run=True,
            bank_id="A1" if normalized_method == "M4" else None,
            output_path=log_root / "dry_runs" / f"{normalized_method.lower()}_predictions.jsonl",
            state_dir=log_root / "state" / setting,
            diagnostics_path=log_root / "dry_runs" / f"{normalized_method.lower()}_diagnostics.json",
            failure_manifest_path=log_root / "dry_runs" / f"{normalized_method.lower()}_failures.jsonl",
        )
    if evaluation == "A1":
        if fold is not None:
            raise LLMProtocolError("A1 must not specify --fold")
        bank_id = "A1" if normalized_method == "M4" else None
        name = "a1_test_predictions.jsonl"
        setting = "a1"
    elif evaluation == "A2" and fold in (1, 2, 3):
        bank_id = f"A2_FOLD_{fold}" if normalized_method == "M4" else None
        name = f"a2_fold_{fold}_test_predictions.jsonl"
        setting = f"a2_fold_{fold}"
    else:
        raise LLMProtocolError("A2 requires --fold 1, 2, or 3")
    method_dir = normalized_method.lower()
    return RunSpec(
        method=normalized_method,
        evaluation=evaluation,
        fold=fold,
        dry_run=False,
        bank_id=bank_id,
        output_path=prediction_root / method_dir / name,
        state_dir=log_root / "state" / method_dir / setting,
        diagnostics_path=log_root / f"{method_dir}_{setting}_diagnostics.json",
        failure_manifest_path=log_root / f"{method_dir}_{setting}_failures.jsonl",
    )


def prepare_run(
    spec: RunSpec,
    *,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    a1_split_path: Path = DEFAULT_A1_SPLIT,
    a2_split_path: Path = DEFAULT_A2_SPLIT,
    config_path: Path = DEFAULT_CONFIG,
    demo_bank_path: Path = DEFAULT_DEMO_BANK,
    m3_prompt_path: Path = DEFAULT_M3_PROMPT,
    m4_prompt_path: Path = DEFAULT_M4_PROMPT,
    ontology_path: Path = DEFAULT_ONTOLOGY,
    review_path: Path = DEFAULT_REVIEW,
    dry_run_count: int = DEFAULT_DRY_RUN_CASES,
) -> dict[str, Any]:
    assert_canonical_input_paths(
        benchmark_path=benchmark_path,
        a1_split_path=a1_split_path,
        a2_split_path=a2_split_path,
        config_path=config_path,
        demo_bank_path=demo_bank_path,
        m3_prompt_path=m3_prompt_path,
        m4_prompt_path=m4_prompt_path,
        ontology_path=ontology_path,
        review_path=review_path,
    )
    validate_canonical_artifact_hashes()
    contract, config, bank = validate_frozen_contract(
        config_path=config_path,
        demo_bank_path=demo_bank_path,
        m3_prompt_path=m3_prompt_path,
        m4_prompt_path=m4_prompt_path,
        ontology_path=ontology_path,
        review_path=review_path,
    )
    benchmark = load_benchmark_index(benchmark_path)
    rows = load_setting_rows(
        "A1" if spec.dry_run else spec.evaluation,
        None if spec.dry_run else spec.fold,
        benchmark,
        a1_path=a1_split_path,
        a2_path=a2_split_path,
    )
    approved_ranks = [int(item["search_rank"]) for item in bank["approved_cases"]]
    if spec.dry_run:
        cases = deterministic_dry_run_cases(
            rows, count=dry_run_count, excluded_ranks=approved_ranks
        )
        heldout: list[str] = []
    else:
        cases = [dict(case) for case in rows if case["role"] == "TEST"]
        if not cases:
            raise LLMProtocolError(f"{spec.setting_id} has no TEST cases")
        heldout = sorted({case["jurisdiction"] for case in cases}) if spec.evaluation == "A2" else []
    demos: list[dict[str, Any]] | None = None
    demo_metadata: dict[str, Any] | None = None
    if spec.method == "M4":
        if spec.bank_id is None:
            raise LLMProtocolError("M4 RunSpec lacks a demo-bank ID")
        demos, demo_metadata = load_demo_bank_for_setting(
            spec.bank_id,
            bank,
            config,
            benchmark,
            actual_test_jurisdictions=heldout,
        )
    return {
        "contract": contract,
        "config": config,
        "demo_bank": bank,
        "benchmark": benchmark,
        "cases": cases,
        "demos": demos,
        "demo_metadata": demo_metadata,
        "heldout_jurisdictions": heldout,
        "split_membership_sha256": target_membership_hash(cases, spec),
        "config_path": config_path,
        "m3_prompt_path": m3_prompt_path,
        "m4_prompt_path": m4_prompt_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("M3", "M4"))
    parser.add_argument("--evaluation", choices=("A1", "A2"), default="A1")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--check-model-access", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-count", type=int, default=DEFAULT_DRY_RUN_CASES)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--base-backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Independent request workers (1--{MAX_WORKERS}; default 1)",
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--demo-bank", type=Path, default=DEFAULT_DEMO_BANK)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser


def _plan(spec: RunSpec, prepared: Mapping[str, Any], log_root: Path) -> dict[str, Any]:
    return {
        "status": "PLAN_VALIDATED_NO_API_CALL",
        "setting_id": spec.setting_id,
        "method": spec.method,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "dry_run": spec.dry_run,
        "case_count": len(prepared["cases"]),
        "search_ranks": [case["search_rank"] for case in prepared["cases"]]
        if spec.dry_run
        else None,
        "all_dry_run_cases_non_test": (
            all(case["role"] != "TEST" for case in prepared["cases"])
            if spec.dry_run
            else None
        ),
        "demo_bank_id": (prepared["demo_metadata"] or {}).get("demo_bank_id"),
        "demo_order": (prepared["demo_metadata"] or {}).get("demo_order", []),
        "heldout_jurisdictions": prepared["heldout_jurisdictions"],
        "output_path": str(spec.output_path),
        "model_access_gate_exists": (log_root / "model_access.json").is_file(),
        "dry_run_gate_exists": dry_run_gate_path(log_root, spec.method).is_file(),
        "split_membership_sha256": prepared["split_membership_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_model_access and args.dry_run:
        raise LLMProtocolError("--check-model-access and --dry-run are mutually exclusive")
    assert_canonical_input_paths(
        benchmark_path=args.benchmark,
        a1_split_path=args.a1_split,
        a2_split_path=args.a2_split,
        config_path=args.config,
        demo_bank_path=args.demo_bank,
    )
    validate_canonical_artifact_hashes()
    contract, _, _ = validate_frozen_contract(
        config_path=args.config,
        demo_bank_path=args.demo_bank,
        m3_prompt_path=DEFAULT_M3_PROMPT,
        m4_prompt_path=DEFAULT_M4_PROMPT,
        ontology_path=DEFAULT_ONTOLOGY,
        review_path=DEFAULT_REVIEW,
    )
    marker_path = args.log_root / "model_access.json"

    if args.check_model_access:
        if args.plan_only:
            print(canonical_json({"status": "MODEL_ACCESS_CHECK_PLANNED", "model": MODEL_ALIAS}))
            return 0
        secret = require_api_key()
        openai_module, sdk_version = load_openai_sdk()
        client = create_openai_client(openai_module, secret)
        marker = perform_model_access_check(
            client,
            sdk_version=sdk_version,
            contract=contract,
            config_path=args.config,
            marker_path=marker_path,
            secret=secret,
        )
        print(
            canonical_json(
                {
                    "status": marker["status"],
                    "effective_model_id": marker["effective_model_id"],
                    "selection_basis": marker["selection_basis"],
                    "marker_path": str(marker_path),
                }
            )
        )
        return 0

    if args.method is None:
        raise LLMProtocolError("--method M3 or --method M4 is required")
    folds: list[int | None]
    if args.dry_run or args.evaluation == "A1":
        if args.fold is not None:
            raise LLMProtocolError("--fold is valid only for A2 execution")
        folds = [None]
    elif args.fold is None:
        folds = [1, 2, 3]
    else:
        folds = [args.fold]

    specs = [
        make_spec(
            args.method,
            "A1" if args.dry_run else args.evaluation,
            fold,
            dry_run=args.dry_run,
            prediction_root=args.prediction_root,
            log_root=args.log_root,
        )
        for fold in folds
    ]
    prepared_runs = [
        prepare_run(
            spec,
            benchmark_path=args.benchmark,
            a1_split_path=args.a1_split,
            a2_split_path=args.a2_split,
            config_path=args.config,
            demo_bank_path=args.demo_bank,
            dry_run_count=args.dry_run_count,
        )
        for spec in specs
    ]
    if args.plan_only:
        print(canonical_json([_plan(spec, prepared, args.log_root) for spec, prepared in zip(specs, prepared_runs)]))
        return 0

    secret = require_api_key()
    openai_module, sdk_version = load_openai_sdk()
    client = create_openai_client(openai_module, secret)
    model_marker = load_model_access_marker(marker_path, contract, sdk_version)
    model_marker = dict(model_marker)
    model_marker["marker_sha256"] = sha256_file(marker_path)
    results: list[dict[str, Any]] = []
    for spec, prepared in zip(specs, prepared_runs):
        if not spec.dry_run:
            validate_dry_run_gate(
                dry_run_gate_path(args.log_root, spec.method),
                spec.method,
                contract=prepared["contract"],
                config=prepared["config"],
                model_marker=model_marker,
            )
        diagnostics = execute_cases(
            spec,
            prepared["cases"],
            client=client,
            secret=secret,
            sdk_version=sdk_version,
            contract=prepared["contract"],
            config=prepared["config"],
            model_marker=model_marker,
            demos=prepared["demos"],
            demo_metadata=prepared["demo_metadata"],
            heldout_jurisdictions=prepared["heldout_jurisdictions"],
            config_path=prepared["config_path"],
            m3_prompt_path=prepared["m3_prompt_path"],
            m4_prompt_path=prepared["m4_prompt_path"],
            max_attempts=args.max_attempts,
            base_backoff_seconds=args.base_backoff_seconds,
            workers=args.workers,
        )
        if spec.dry_run:
            write_dry_run_gate(
                dry_run_gate_path(args.log_root, spec.method),
                spec,
                diagnostics,
                prepared["cases"],
                contract=prepared["contract"],
                config=prepared["config"],
                model_marker=model_marker,
                demo_metadata=prepared["demo_metadata"],
                secret=secret,
            )
        results.append(diagnostics)
        if diagnostics["status"] not in {"COMPLETE"}:
            break
    print(canonical_json(results))
    return 0 if results and all(item["status"] == "COMPLETE" for item in results) else 1


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except LLMProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
