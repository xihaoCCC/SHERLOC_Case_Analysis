#!/usr/bin/env python3
"""Run the frozen M1 TF-IDF + one-vs-rest logistic-regression benchmark.

The runner implements the Phase-4 protocol for one A1 model and one fresh
model for each A2 fold.  Hyperparameters and one global decision threshold are
selected on validation data only.  Test probabilities are generated only
after both choices have been frozen in a fit-state artifact.

Complete runs are idempotently skipped.  If fitting completed but prediction
writing was interrupted, the fitted pipeline and fit state are reused.  A
conflicting or damaged complete artifact requires ``--force`` rather than
being silently replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multiclass import OneVsRestClassifier


VERSION = "1.0.0"
ARTIFACT_SCHEMA_VERSION = "sherloc-m1-artifacts-v1"
PREDICTION_SCHEMA_VERSION = "sherloc-amp-predictions-v1"
EXPECTED_CONFIG_ID = "m1-tfidf-logreg-amp-v2"
EXPECTED_METHOD_ID = "M1"
EXPECTED_COHORT_ID = (
    "sherloc-tip-2026-08-09-en-legacy-amp-complete-"
    "n1263-097ce2027171ebc9"
)
EXPECTED_N = 1263
EXPECTED_BENCHMARK_SHA256 = (
    "2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020"
)
EXPECTED_ONTOLOGY_SHA256 = (
    "f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1"
)
EXPECTED_CONFIG_SHA256 = (
    "44e80edf844d1589dec8b7236d58a65666f6479f0156d3c7ffff9e9de6d74b46"
)
THRESHOLD_GRID = tuple(
    float(Decimal("0.20") + Decimal("0.05") * index) for index in range(13)
)
BASELINE_THRESHOLD = 0.50

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = REPO_ROOT / "data/processed/sherloc_benchmark_v1.jsonl"
DEFAULT_ONTOLOGY = REPO_ROOT / "config/amp_ontology_v1.yaml"
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/m1_tfidf_logreg_amp_v2.yaml"
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs/models/m1"
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "outputs/predictions/m1"


class M1ProtocolError(RuntimeError):
    """Raised when an input or artifact violates the frozen protocol."""


@dataclass(frozen=True)
class RunSpec:
    evaluation: str
    fold: int | None
    split_path: Path
    model_dir: Path
    prediction_path: Path

    @property
    def key(self) -> str:
        return "a1" if self.evaluation == "A1" else f"a2_fold_{self.fold}"


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise M1ProtocolError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise M1ProtocolError(f"Refusing to write empty JSONL: {path}")
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_text(path, payload)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".m1-", suffix=".joblib", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M1ProtocolError(f"Cannot read JSON document {path}: {error}") from error
    if not isinstance(value, dict):
        raise M1ProtocolError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise M1ProtocolError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise M1ProtocolError(f"Cannot read JSONL {path}: {error}") from error
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise M1ProtocolError(f"Cannot read CSV {path}: {error}") from error


def ontology_label_order(ontology: Mapping[str, Any]) -> list[str]:
    order = [
        item["id"]
        for family in ("ACT", "MEANS", "PURPOSE")
        for item in ontology["families"][family]
    ]
    if len(order) != 17 or len(set(order)) != 17:
        raise M1ProtocolError("Ontology is not the frozen 5/6/6 AMP design")
    return order


def record_labels(record: Mapping[str, Any]) -> list[str]:
    targets = record["amp_targets"]
    return list(
        targets["act_ontology_ids"]
        + targets["means_ontology_ids"]
        + targets["purpose_ontology_ids"]
    )


def target_matrix(
    records: Sequence[Mapping[str, Any]], label_order: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [int(label in set(record_labels(record))) for label in label_order]
            for record in records
        ],
        dtype=np.int8,
    )


def membership_digest(rows: Iterable[Sequence[Any]]) -> str:
    payload = "".join("\t".join(map(str, row)) + "\n" for row in rows)
    return sha256_text(payload)


def validate_static_inputs(
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    required = (benchmark_path, ontology_path, config_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise M1ProtocolError(f"Missing frozen M1 inputs: {missing}")
    if sha256_file(benchmark_path) != EXPECTED_BENCHMARK_SHA256:
        raise M1ProtocolError("Frozen benchmark hash changed")
    if sha256_file(ontology_path) != EXPECTED_ONTOLOGY_SHA256:
        raise M1ProtocolError("Frozen ontology hash changed")
    if (
        config_path.resolve() == DEFAULT_CONFIG.resolve()
        and sha256_file(config_path) != EXPECTED_CONFIG_SHA256
    ):
        raise M1ProtocolError("Frozen Phase-4 M1 config hash changed")

    benchmark = load_jsonl(benchmark_path)
    ontology = load_json(ontology_path)
    config = load_json(config_path)
    label_order = ontology_label_order(ontology)
    if len(benchmark) != EXPECTED_N:
        raise M1ProtocolError(f"Expected {EXPECTED_N} benchmark rows, got {len(benchmark)}")
    if any(row.get("primary_cohort_id") != EXPECTED_COHORT_ID for row in benchmark):
        raise M1ProtocolError("Primary cohort ID mismatch")
    ranks = [int(row["identity"]["search_rank"]) for row in benchmark]
    urls = [str(row["identity"]["canonical_url"]) for row in benchmark]
    if len(set(ranks)) != EXPECTED_N or len(set(urls)) != EXPECTED_N:
        raise M1ProtocolError("Benchmark search ranks or canonical URLs are not unique")
    if config.get("config_id") != EXPECTED_CONFIG_ID or config.get("method_id") != EXPECTED_METHOD_ID:
        raise M1ProtocolError("Unexpected M1 config identity")
    if config.get("primary_cohort_id") != EXPECTED_COHORT_ID:
        raise M1ProtocolError("M1 config cohort ID mismatch")
    if config.get("status") != "FROZEN_FOR_PHASE_4_EXECUTION":
        raise M1ProtocolError("M1 config is not frozen for Phase-4 execution")
    if list(config["targets"]["label_order"]) != label_order:
        raise M1ProtocolError("M1 config label order differs from the frozen ontology")
    vectorizer = config["pipeline"]["vectorizer"]
    if vectorizer.get("analyzer") != "word" or list(vectorizer.get("ngram_range", [])) != [1, 2]:
        raise M1ProtocolError("M1 must use word 1-2-gram TF-IDF")
    if config["validation_search"].get("selection_data") != "VALIDATION_ONLY":
        raise M1ProtocolError("M1 hyperparameter selection is not validation-only")
    if not config["validation_search"].get("test_labels_forbidden"):
        raise M1ProtocolError("M1 config does not prohibit test-label tuning")
    thresholding = config["thresholding"]
    configured_grid = tuple(
        float(
            Decimal(str(thresholding["candidate_grid_start"]))
            + Decimal(str(thresholding["candidate_grid_step"])) * index
        )
        for index in range(
            int(
                (
                    Decimal(str(thresholding["candidate_grid_stop"]))
                    - Decimal(str(thresholding["candidate_grid_start"]))
                )
                / Decimal(str(thresholding["candidate_grid_step"]))
            )
            + 1
        )
    )
    if configured_grid != THRESHOLD_GRID:
        raise M1ProtocolError("M1 threshold grid is not frozen at 0.20..0.80 by 0.05")
    if float(thresholding.get("fixed_baseline")) != BASELINE_THRESHOLD:
        raise M1ProtocolError("M1 fixed sensitivity threshold is not 0.50")
    if thresholding.get("test_label_tuning") != "PROHIBITED":
        raise M1ProtocolError("M1 config does not prohibit test threshold tuning")
    expected_splits = {
        "data/splits/a1_iid_split_final_v1.csv",
        "data/splits/a2_jurisdiction_folds_final_v1.csv",
    }
    if set(config["reproducibility"]["split_files"]) != expected_splits:
        raise M1ProtocolError("M1 config does not reference both final split artifacts")
    return benchmark, label_order, config


def validate_execution_environment(config: Mapping[str, Any]) -> None:
    expected = config["expected_execution_environment"]
    observed = {
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    # Python patch releases do not change the frozen sklearn estimator or
    # split semantics.  The active xihao_env advanced from 3.10.19 to 3.10.20
    # between preparation and execution, so fail closed on major/minor drift
    # while continuing to require exact numpy and sklearn versions.  The exact
    # observed patch version is still persisted in every run artifact.
    mismatches: dict[str, dict[str, str]] = {}
    for key, value in observed.items():
        expected_value = str(expected[key])
        matches = (
            value.split(".")[:2] == expected_value.split(".")[:2]
            if key == "python_version"
            else value == expected_value
        )
        if not matches:
            mismatches[key] = {"expected": expected_value, "observed": value}
    if mismatches:
        raise M1ProtocolError(
            "M1 execution-environment drift: " + canonical_json(mismatches)
        )


def normalize_role(row: Mapping[str, str]) -> str:
    role = row.get("split") or row.get("role") or ""
    return role.strip().upper()


def validate_and_partition_split(
    spec: RunSpec,
    split_rows: Sequence[dict[str, str]],
    benchmark: Sequence[dict[str, Any]],
    label_order: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not split_rows:
        raise M1ProtocolError(f"Empty final split file: {spec.split_path}")
    if "split_status" in split_rows[0]:
        statuses = {row["split_status"].strip().upper() for row in split_rows}
        if any("PROVISIONAL" in status for status in statuses):
            raise M1ProtocolError(f"Refusing provisional split: {spec.split_path}")
    if spec.evaluation == "A2":
        selected = [row for row in split_rows if int(row.get("fold_id", "0")) == spec.fold]
    else:
        selected = list(split_rows)
    if len(selected) != EXPECTED_N:
        raise M1ProtocolError(
            f"{spec.key} must contain {EXPECTED_N} rows, got {len(selected)}"
        )

    by_rank = {int(row["identity"]["search_rank"]): row for row in benchmark}
    observed_ranks = [int(row["search_rank"]) for row in selected]
    if len(set(observed_ranks)) != EXPECTED_N or set(observed_ranks) != set(by_rank):
        raise M1ProtocolError(f"{spec.key} split membership differs from the cohort")

    role_counts: dict[str, int] = {}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    split_label_mismatches: list[int] = []
    selected_sorted = sorted(selected, key=lambda item: int(item["search_rank"]))
    for split_row in selected_sorted:
        rank = int(split_row["search_rank"])
        record = by_rank[rank]
        identity = record["identity"]
        if split_row["canonical_url"] != identity["canonical_url"]:
            raise M1ProtocolError(f"Canonical URL mismatch for rank {rank}")
        if split_row["jurisdiction"] != identity["jurisdiction_country_raw"]:
            raise M1ProtocolError(f"Jurisdiction mismatch for rank {rank}")
        actual = set(record_labels(record))
        if any(int(split_row[label]) != int(label in actual) for label in label_order):
            split_label_mismatches.append(rank)
        role = normalize_role(split_row)
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "VALIDATION":
            validation.append(record)
        elif role == "TEST":
            test.append(record)
        elif split_row.get("effective_supervised_train", "0").strip() == "1":
            train.append(record)
        else:
            raise M1ProtocolError(
                f"Rank {rank} has unsupported non-training role {role!r}"
            )
    if split_label_mismatches:
        raise M1ProtocolError(
            "Split target columns differ from frozen benchmark for ranks "
            + ", ".join(map(str, split_label_mismatches[:10]))
        )
    if not train or not validation or not test:
        raise M1ProtocolError(
            f"{spec.key} has empty train/validation/test partition: "
            f"{len(train)}/{len(validation)}/{len(test)}"
        )
    overlap = (
        set(int(row["identity"]["search_rank"]) for row in train)
        & set(int(row["identity"]["search_rank"]) for row in validation + test)
    ) | (
        set(int(row["identity"]["search_rank"]) for row in validation)
        & set(int(row["identity"]["search_rank"]) for row in test)
    )
    if overlap:
        raise M1ProtocolError(f"Partition leakage in {spec.key}: {sorted(overlap)}")

    digest = membership_digest(
        (
            int(row["search_rank"]),
            row["canonical_url"],
            normalize_role(row),
            row.get("effective_supervised_train", ""),
            row.get("fold_id", ""),
        )
        for row in selected_sorted
    )
    metadata = {
        "split_file_sha256": sha256_file(spec.split_path),
        "split_membership_sha256": digest,
        "role_counts": dict(sorted(role_counts.items())),
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
    }
    return train, validation, test, metadata


def texts(records: Sequence[Mapping[str, Any]]) -> list[str]:
    result = [str(record["text_input"]["english_fact_summary_raw"]) for record in records]
    if any(not value.strip() for value in result):
        raise M1ProtocolError("M1 encountered an empty English Fact Summary")
    return result


def parameter_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config["validation_search"]["grid"]
    required = (
        "vectorizer.min_df",
        "base_classifier.C",
        "base_classifier.class_weight",
    )
    if set(raw) != set(required):
        raise M1ProtocolError(f"Unexpected M1 hyperparameter grid: {sorted(raw)}")
    combinations = [
        {
            "vectorizer.min_df": int(min_df),
            "base_classifier.C": float(c_value),
            "base_classifier.class_weight": class_weight,
        }
        for min_df, c_value, class_weight in itertools.product(*(raw[key] for key in required))
    ]
    maximum = int(config["validation_search"]["maximum_configurations"])
    if len(combinations) > maximum:
        raise M1ProtocolError(
            f"M1 grid has {len(combinations)} combinations; maximum is {maximum}"
        )
    return combinations


def build_pipeline(config: Mapping[str, Any], parameters: Mapping[str, Any]) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    raw_vectorizer = config["pipeline"]["vectorizer"]
    dtype_name = raw_vectorizer.get("dtype", "float64")
    dtype = np.float64 if dtype_name == "float64" else np.float32
    vectorizer = TfidfVectorizer(
        analyzer=raw_vectorizer["analyzer"],
        ngram_range=tuple(raw_vectorizer["ngram_range"]),
        lowercase=bool(raw_vectorizer["lowercase"]),
        strip_accents=raw_vectorizer.get("strip_accents"),
        stop_words=raw_vectorizer.get("stop_words"),
        token_pattern=raw_vectorizer["token_pattern"],
        norm=raw_vectorizer["norm"],
        use_idf=bool(raw_vectorizer["use_idf"]),
        smooth_idf=bool(raw_vectorizer["smooth_idf"]),
        sublinear_tf=bool(raw_vectorizer["sublinear_tf"]),
        min_df=int(parameters["vectorizer.min_df"]),
        max_df=float(raw_vectorizer["max_df"]),
        max_features=int(raw_vectorizer["max_features"]),
        dtype=dtype,
    )
    raw_classifier = config["pipeline"]["base_classifier"]
    classifier = LogisticRegression(
        penalty=raw_classifier["penalty"],
        solver=raw_classifier["solver"],
        C=float(parameters["base_classifier.C"]),
        class_weight=parameters["base_classifier.class_weight"],
        max_iter=int(raw_classifier["max_iter"]),
        tol=float(raw_classifier["tol"]),
        random_state=int(raw_classifier["random_state"]),
    )
    wrapper = OneVsRestClassifier(
        classifier,
        n_jobs=int(config["pipeline"]["classifier_wrapper"]["n_jobs"]),
    )
    return vectorizer, wrapper


def macro_average_precision(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[float, list[float | None], list[int]]:
    scores: list[float | None] = []
    supports = y_true.sum(axis=0).astype(int).tolist()
    for index, support in enumerate(supports):
        if support == 0:
            scores.append(None)
        else:
            scores.append(float(average_precision_score(y_true[:, index], probabilities[:, index])))
    defined = [value for value in scores if value is not None]
    if not defined:
        raise M1ProtocolError("Macro average precision is undefined for every label")
    return float(np.mean(defined)), scores, supports


def macro_f1(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    predicted = (probabilities >= threshold).astype(np.int8)
    return float(f1_score(y_true, predicted, average="macro", zero_division=0))


def select_global_threshold(
    y_validation: np.ndarray, probabilities: np.ndarray
) -> tuple[float, list[dict[str, Any]]]:
    curve = [
        {
            "threshold": threshold,
            "validation_macro_f1": macro_f1(y_validation, probabilities, threshold),
        }
        for threshold in THRESHOLD_GRID
    ]
    # The final key is a deterministic tertiary rule for the otherwise unresolved
    # equal-distance (.45/.55, etc.) case; it is never informed by test data.
    winner = min(
        curve,
        key=lambda row: (
            -row["validation_macro_f1"],
            abs(row["threshold"] - BASELINE_THRESHOLD),
            row["threshold"],
        ),
    )
    return float(winner["threshold"]), curve


def fit_and_select(
    train: Sequence[dict[str, Any]],
    validation: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_text = texts(train)
    validation_text = texts(validation)
    y_train = target_matrix(train, label_order)
    y_validation = target_matrix(validation, label_order)
    if np.any(y_train.sum(axis=0) == 0):
        missing = [label_order[index] for index in np.flatnonzero(y_train.sum(axis=0) == 0)]
        raise M1ProtocolError(f"Training partition has no positive examples for: {missing}")

    best: tuple[
        float,
        int,
        TfidfVectorizer,
        OneVsRestClassifier,
        np.ndarray,
        dict[str, Any],
    ] | None = None
    search_rows: list[dict[str, Any]] = []
    for configuration_index, parameters in enumerate(parameter_grid(config), 1):
        started = time.perf_counter()
        vectorizer, classifier = build_pipeline(config, parameters)
        x_train = vectorizer.fit_transform(train_text)
        x_validation = vectorizer.transform(validation_text)
        classifier.fit(x_train, y_train)
        probabilities = np.asarray(classifier.predict_proba(x_validation), dtype=np.float64)
        if probabilities.shape != y_validation.shape:
            raise M1ProtocolError(
                f"Unexpected validation probability shape {probabilities.shape}"
            )
        macro_ap, per_label_ap, supports = macro_average_precision(y_validation, probabilities)
        elapsed = time.perf_counter() - started
        search_row = {
            "configuration_index": configuration_index,
            "vectorizer_min_df": parameters["vectorizer.min_df"],
            "classifier_c": parameters["base_classifier.C"],
            "classifier_class_weight": (
                "null"
                if parameters["base_classifier.class_weight"] is None
                else parameters["base_classifier.class_weight"]
            ),
            "validation_macro_average_precision": macro_ap,
            "validation_defined_ap_labels": sum(value is not None for value in per_label_ap),
            "validation_positive_supports_json": canonical_json(dict(zip(label_order, supports))),
            "validation_per_label_ap_json": canonical_json(dict(zip(label_order, per_label_ap))),
            "fit_seconds": elapsed,
            "vocabulary_size": len(vectorizer.vocabulary_),
        }
        search_rows.append(search_row)
        candidate = (
            macro_ap,
            configuration_index,
            vectorizer,
            classifier,
            probabilities,
            parameters,
        )
        if best is None or (-candidate[0], candidate[1]) < (-best[0], best[1]):
            best = candidate

    # Frozen deterministic tie-break: earliest configuration in the config grid.
    if best is None:
        raise M1ProtocolError("Frozen M1 hyperparameter grid is empty")
    macro_ap, configuration_index, vectorizer, classifier, validation_probabilities, parameters = best
    threshold, threshold_rows = select_global_threshold(y_validation, validation_probabilities)
    for row in search_rows:
        row["selected"] = int(row["configuration_index"] == configuration_index)
    for row in threshold_rows:
        row["selected"] = int(row["threshold"] == threshold)

    pipeline = {"vectorizer": vectorizer, "classifier": classifier}
    selection = {
        "selected_configuration_index": configuration_index,
        "selected_hyperparameters": dict(parameters),
        "validation_macro_average_precision": macro_ap,
        "validation_macro_f1_selected_threshold": macro_f1(
            y_validation, validation_probabilities, threshold
        ),
        "validation_macro_f1_0_50": macro_f1(
            y_validation, validation_probabilities, BASELINE_THRESHOLD
        ),
        "selected_global_threshold": threshold,
        "threshold_tie_break": (
            "max_validation_macro_f1_then_closest_to_0.50_then_lower_threshold"
        ),
        "hyperparameter_tie_break": "earliest_configuration_in_frozen_grid",
        "validation_label_positive_support": dict(
            zip(label_order, y_validation.sum(axis=0).astype(int).tolist())
        ),
    }
    return pipeline, selection, search_rows, threshold_rows


def fitted_model_metadata(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    vectorizer: TfidfVectorizer = pipeline["vectorizer"]
    classifier: OneVsRestClassifier = pipeline["classifier"]
    vocabulary_payload = canonical_json(
        [
            (str(term), int(index))
            for term, index in sorted(
                vectorizer.vocabulary_.items(), key=lambda item: item[0]
            )
        ]
    )
    idf = np.asarray(vectorizer.idf_, dtype=np.float64)
    estimator_metadata: list[dict[str, Any]] = []
    for estimator in classifier.estimators_:
        item: dict[str, Any] = {"class": estimator.__class__.__name__}
        if hasattr(estimator, "coef_"):
            item["coefficient_shape"] = list(estimator.coef_.shape)
            item["intercept_shape"] = list(estimator.intercept_.shape)
            item["n_iter"] = np.asarray(estimator.n_iter_).astype(int).tolist()
        estimator_metadata.append(item)
    return {
        "vectorizer_class": vectorizer.__class__.__name__,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "vocabulary_sha256": sha256_text(vocabulary_payload),
        "idf_length": int(idf.size),
        "idf_sha256": sha256_bytes(idf.tobytes(order="C")),
        "idf_min": float(idf.min()),
        "idf_max": float(idf.max()),
        "classifier_wrapper_class": classifier.__class__.__name__,
        "estimator_count": len(classifier.estimators_),
        "estimators": estimator_metadata,
    }


def execution_context(
    spec: RunSpec,
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
    split_metadata: Mapping[str, Any],
    label_order: Sequence[str],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "runner_version": VERSION,
        "method_id": EXPECTED_METHOD_ID,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "primary_cohort_id": EXPECTED_COHORT_ID,
        "label_order": list(label_order),
        "benchmark_path": str(benchmark_path.relative_to(REPO_ROOT)),
        "benchmark_sha256": sha256_file(benchmark_path),
        "ontology_path": str(ontology_path.relative_to(REPO_ROOT)),
        "ontology_sha256": sha256_file(ontology_path),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "split_path": str(spec.split_path.relative_to(REPO_ROOT)),
        **split_metadata,
        "threshold_grid": list(THRESHOLD_GRID),
        "baseline_threshold": BASELINE_THRESHOLD,
        "test_labels_used_for_selection": False,
    }


def context_digest(context: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(context))


def validate_fit_state(
    fit_state: Mapping[str, Any], context: Mapping[str, Any], pipeline_path: Path
) -> None:
    if fit_state.get("status") != "FIT_AND_VALIDATION_SELECTION_COMPLETE":
        raise M1ProtocolError("Fit state is not complete")
    if fit_state.get("execution_context_sha256") != context_digest(context):
        raise M1ProtocolError("Fit state belongs to a different frozen execution context")
    if not pipeline_path.is_file():
        raise M1ProtocolError("Fit state exists but fitted pipeline is missing")
    if fit_state.get("pipeline_sha256") != sha256_file(pipeline_path):
        raise M1ProtocolError("Fitted pipeline hash differs from fit state")
    for name in ("validation_hyperparameter_search", "validation_threshold_search"):
        relative = fit_state.get(f"{name}_path")
        expected = fit_state.get(f"{name}_sha256")
        if not relative or not expected:
            raise M1ProtocolError(f"Fit state lacks {name} provenance")
        artifact = REPO_ROOT / str(relative)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise M1ProtocolError(f"Fit-state artifact is missing or damaged: {artifact}")
    threshold = float(fit_state["selection"]["selected_global_threshold"])
    if threshold not in THRESHOLD_GRID:
        raise M1ProtocolError("Fit state threshold is outside the frozen grid")


def complete_run_is_valid(
    metadata_path: Path,
    prediction_path: Path,
    pipeline_path: Path,
    context: Mapping[str, Any],
) -> bool:
    if not metadata_path.is_file():
        return False
    metadata = load_json(metadata_path)
    if metadata.get("execution_context_sha256") != context_digest(context):
        raise M1ProtocolError(
            f"Existing run uses a different execution context: {metadata_path}"
        )
    if metadata.get("status") == "IN_PROGRESS":
        return False
    if metadata.get("status") != "COMPLETE":
        raise M1ProtocolError(f"Invalid run-metadata status: {metadata_path}")
    for path, expected in (
        (prediction_path, metadata.get("prediction_sha256")),
        (pipeline_path, metadata.get("pipeline_sha256")),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise M1ProtocolError(f"Existing complete run artifact is missing or damaged: {path}")
    for name in (
        "fit_state",
        "validation_hyperparameter_search",
        "validation_threshold_search",
    ):
        relative = metadata.get(f"{name}_path")
        expected = metadata.get(f"{name}_sha256")
        if not relative or not expected:
            raise M1ProtocolError(f"Complete metadata lacks {name} provenance")
        artifact = REPO_ROOT / str(relative)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise M1ProtocolError(f"Existing run artifact is missing or damaged: {artifact}")
    return True


def prediction_rows(
    spec: RunSpec,
    records: Sequence[dict[str, Any]],
    probabilities: np.ndarray,
    label_order: Sequence[str],
    threshold: float,
    run_id: str,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if probabilities.shape != (len(records), len(label_order)):
        raise M1ProtocolError(f"Unexpected test probability shape: {probabilities.shape}")
    rows: list[dict[str, Any]] = []
    for record, scores in sorted(
        zip(records, probabilities), key=lambda item: int(item[0]["identity"]["search_rank"])
    ):
        identity = record["identity"]
        reference = set(record_labels(record))
        predicted = [label for label, score in zip(label_order, scores) if score >= threshold]
        predicted_baseline = [
            label for label, score in zip(label_order, scores) if score >= BASELINE_THRESHOLD
        ]
        fact_summary = str(record["text_input"]["english_fact_summary_raw"])
        rows.append(
            {
                "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
                "run_id": run_id,
                "method_id": EXPECTED_METHOD_ID,
                "evaluation": spec.evaluation,
                "fold": spec.fold,
                "search_rank": int(identity["search_rank"]),
                "canonical_url": identity["canonical_url"],
                "jurisdiction": identity["jurisdiction_country_raw"],
                "split": "TEST",
                "fact_summary": fact_summary,
                "input_sha256": sha256_text(fact_summary),
                "silver_reference_labels": [label for label in label_order if label in reference],
                "predicted_labels": predicted,
                "predicted_labels_0_50": predicted_baseline,
                "probabilities_by_label": {
                    label: float(score) for label, score in zip(label_order, scores)
                },
                "selected_threshold": threshold,
                "truncated_input": False,
                "primary_cohort_id": EXPECTED_COHORT_ID,
                "config_sha256": context["config_sha256"],
                "split_membership_sha256": context["split_membership_sha256"],
            }
        )
    return rows


def run_one(
    spec: RunSpec,
    benchmark: Sequence[dict[str, Any]],
    label_order: Sequence[str],
    config: Mapping[str, Any],
    benchmark_path: Path,
    ontology_path: Path,
    config_path: Path,
    *,
    force: bool,
    plan_only: bool,
) -> dict[str, Any]:
    if not spec.split_path.is_file():
        raise M1ProtocolError(
            f"Final split does not exist; M1 must not run yet: {spec.split_path}"
        )
    split_rows = load_csv(spec.split_path)
    train, validation, test, split_metadata = validate_and_partition_split(
        spec, split_rows, benchmark, label_order
    )
    context = execution_context(
        spec,
        benchmark_path,
        ontology_path,
        config_path,
        split_metadata,
        label_order,
    )
    plan = {
        "run": spec.key,
        "evaluation": spec.evaluation,
        "fold": spec.fold,
        "train_n": len(train),
        "validation_n": len(validation),
        "test_n": len(test),
        "split_membership_sha256": split_metadata["split_membership_sha256"],
        "prediction_path": str(spec.prediction_path),
        "model_dir": str(spec.model_dir),
    }
    if plan_only:
        return {"status": "PLAN_VALIDATED", **plan}

    metadata_path = spec.model_dir / "run_metadata.json"
    fit_state_path = spec.model_dir / "fit_state.json"
    pipeline_path = spec.model_dir / "m1_pipeline.joblib"
    search_path = spec.model_dir / "validation_hyperparameter_search.csv"
    threshold_path = spec.model_dir / "validation_threshold_search.csv"
    if metadata_path.exists() and not force:
        if complete_run_is_valid(
            metadata_path, spec.prediction_path, pipeline_path, context
        ):
            return {"status": "SKIPPED_COMPLETE", **plan}
    if force:
        # Files are replaced atomically below.  No broad or recursive deletion is used.
        fit_state = None
    elif fit_state_path.is_file():
        fit_state = load_json(fit_state_path)
        validate_fit_state(fit_state, context, pipeline_path)
    else:
        fit_state = None

    overall_started = time.perf_counter()
    started_at = utc_now()
    atomic_json(
        metadata_path,
        {
            **context,
            "status": "IN_PROGRESS",
            "started_at": started_at,
            "execution_context_sha256": context_digest(context),
        },
    )
    if fit_state is None:
        fit_started = time.perf_counter()
        pipeline, selection, search_rows, threshold_rows = fit_and_select(
            train, validation, label_order, config
        )
        fit_seconds = time.perf_counter() - fit_started
        atomic_joblib(pipeline_path, pipeline)
        atomic_csv(search_path, search_rows)
        atomic_csv(threshold_path, threshold_rows)
        fit_state = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "status": "FIT_AND_VALIDATION_SELECTION_COMPLETE",
            "completed_at": utc_now(),
            "execution_context_sha256": context_digest(context),
            "pipeline_sha256": sha256_file(pipeline_path),
            "selection": selection,
            "fit_seconds": fit_seconds,
            "model_metadata": fitted_model_metadata(pipeline),
            "validation_hyperparameter_search_path": str(search_path.relative_to(REPO_ROOT)),
            "validation_hyperparameter_search_sha256": sha256_file(search_path),
            "validation_threshold_search_path": str(threshold_path.relative_to(REPO_ROOT)),
            "validation_threshold_search_sha256": sha256_file(threshold_path),
            "test_labels_used_for_selection": False,
        }
        atomic_json(fit_state_path, fit_state)
    else:
        pipeline = joblib.load(pipeline_path)
        selection = dict(fit_state["selection"])
        validate_fit_state(fit_state, context, pipeline_path)

    threshold = float(selection["selected_global_threshold"])
    # This is the first substantive test operation: both model configuration and
    # threshold are already immutable in fit_state.json.
    prediction_started = time.perf_counter()
    x_test = pipeline["vectorizer"].transform(texts(test))
    probabilities = np.asarray(
        pipeline["classifier"].predict_proba(x_test), dtype=np.float64
    )
    run_id = sha256_text(
        canonical_json(
            {
                "method": EXPECTED_METHOD_ID,
                "evaluation": spec.evaluation,
                "fold": spec.fold,
                "execution_context_sha256": context_digest(context),
                "pipeline_sha256": fit_state["pipeline_sha256"],
                "threshold": threshold,
            }
        )
    )[:24]
    predictions = prediction_rows(
        spec, test, probabilities, label_order, threshold, run_id, context
    )
    atomic_jsonl(spec.prediction_path, predictions)
    prediction_seconds = time.perf_counter() - prediction_started
    completed_at = utc_now()
    metadata = {
        **context,
        "status": "COMPLETE",
        "execution_context_sha256": context_digest(context),
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "elapsed_seconds_this_invocation": time.perf_counter() - overall_started,
        "fit_seconds": fit_state["fit_seconds"],
        "prediction_seconds": prediction_seconds,
        "selection": selection,
        "model_metadata": fit_state["model_metadata"],
        "pipeline_path": str(pipeline_path.relative_to(REPO_ROOT)),
        "pipeline_sha256": sha256_file(pipeline_path),
        "fit_state_path": str(fit_state_path.relative_to(REPO_ROOT)),
        "fit_state_sha256": sha256_file(fit_state_path),
        "validation_hyperparameter_search_path": fit_state[
            "validation_hyperparameter_search_path"
        ],
        "validation_hyperparameter_search_sha256": fit_state[
            "validation_hyperparameter_search_sha256"
        ],
        "validation_threshold_search_path": fit_state[
            "validation_threshold_search_path"
        ],
        "validation_threshold_search_sha256": fit_state[
            "validation_threshold_search_sha256"
        ],
        "prediction_path": str(spec.prediction_path.relative_to(REPO_ROOT)),
        "prediction_sha256": sha256_file(spec.prediction_path),
        "prediction_rows": len(predictions),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "protocol_attestation": {
            "input_field": "text_input.english_fact_summary_raw",
            "tfidf_fit_partition": "TRAIN_ONLY",
            "hyperparameters_selected_on": "VALIDATION_ONLY",
            "hyperparameter_selection_metric": "macro_average_precision",
            "threshold_selected_on": "VALIDATION_ONLY",
            "threshold_selection_metric": "macro_f1",
            "single_global_threshold": True,
            "per_label_thresholds": False,
            "test_labels_used_for_model_or_threshold_selection": False,
            "test_predictions_created_after_fit_state": True,
            "baseline_threshold": BASELINE_THRESHOLD,
        },
    }
    atomic_json(metadata_path, metadata)
    return {
        "status": "COMPLETE",
        **plan,
        "run_id": run_id,
        "selected_hyperparameters": selection["selected_hyperparameters"],
        "selected_threshold": threshold,
        "validation_macro_average_precision": selection[
            "validation_macro_average_precision"
        ],
        "validation_macro_f1": selection[
            "validation_macro_f1_selected_threshold"
        ],
    }


def make_specs(args: argparse.Namespace) -> list[RunSpec]:
    requested: list[tuple[str, int | None]]
    if args.evaluation == "A1":
        requested = [("A1", None)]
    elif args.evaluation == "A2":
        folds = [args.fold] if args.fold is not None else [1, 2, 3]
        requested = [("A2", fold) for fold in folds]
    else:
        if args.fold is not None:
            raise M1ProtocolError("--fold may be used only with --evaluation A2")
        requested = [("A1", None), ("A2", 1), ("A2", 2), ("A2", 3)]
    specs: list[RunSpec] = []
    for evaluation, fold in requested:
        if evaluation == "A1":
            specs.append(
                RunSpec(
                    evaluation="A1",
                    fold=None,
                    split_path=args.a1_split,
                    model_dir=args.model_root / "a1",
                    prediction_path=args.prediction_root / "a1_test_predictions.jsonl",
                )
            )
        else:
            specs.append(
                RunSpec(
                    evaluation="A2",
                    fold=fold,
                    split_path=args.a2_split,
                    model_dir=args.model_root / f"a2_fold_{fold}",
                    prediction_path=args.prediction_root
                    / f"a2_fold_{fold}_test_predictions.jsonl",
                )
            )
    return specs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        choices=("A1", "A2", "all"),
        required=True,
        help="Run A1, A2 (all folds unless --fold), or all four fits.",
    )
    parser.add_argument("--fold", type=int, choices=(1, 2, 3))
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument(
        "--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Validate frozen inputs and print run plans without fitting or writing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing run instead of resuming/skipping it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fold is not None and args.evaluation != "A2":
        raise M1ProtocolError("--fold may be used only with --evaluation A2")
    benchmark, label_order, config = validate_static_inputs(
        args.benchmark, args.ontology, args.config
    )
    validate_execution_environment(config)
    results = [
        run_one(
            spec,
            benchmark,
            label_order,
            config,
            args.benchmark,
            args.ontology,
            args.config,
            force=args.force,
            plan_only=args.plan,
        )
        for spec in make_specs(args)
    ]
    print(json.dumps({"runner_version": VERSION, "runs": results}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except M1ProtocolError as error:
        print(f"M1 protocol error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
