#!/usr/bin/env python3
"""Run the dedicated leakage-free fixed-protocol Evaluation B M1 model.

The default-safe ``--preflight`` mode validates the frozen human membership,
writes the common exclusion audit, and does not fit a model. ``--execute`` is
additionally gated by an explicit QC/membership-freeze confirmation. No
hyperparameter or threshold search exists in this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from evaluation_b_supervised import (
    DEFAULT_A1_SPLIT,
    DEFAULT_A2_SPLIT,
    DEFAULT_AUDIT,
    DEFAULT_BENCHMARK,
    DEFAULT_HUMAN_REFERENCE,
    DEFAULT_MEMBERSHIP,
    DEFAULT_MEMBERSHIP_FREEZE,
    DEFAULT_ONTOLOGY,
    DEFAULT_PREFLIGHT,
    DEFAULT_RELIABILITY_SAMPLE,
    EvaluationBSupervisedError,
    RunLock,
    atomic_json,
    atomic_jsonl,
    canonical_json,
    prediction_rows,
    prepare_supervised_data,
    sha256_file,
    target_matrix,
    texts,
    training_membership_sha256,
    utc_now,
    write_preflight_artifacts,
)


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config/experiments/eval_b_m1_tfidf_logreg_v1.yaml"
DEFAULT_M2_CONFIG = REPO_ROOT / "config/experiments/eval_b_m2_modernbert_v1.yaml"
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs/models/evaluation_b/m1"
DEFAULT_PREDICTION = REPO_ROOT / "outputs/predictions/evaluation_b/m1/predictions.jsonl"
EXPECTED_CONFIG_SHA256 = "5c6a916af3781305926b0cd57bde77e30f7c094a035a313cda95fc391a4046a5"
EXPECTED_M2_CONFIG_SHA256 = "5a83104cda51b8674ab577ea133be991ebccab99a30915e3fc307b219a64ed7b"
FIXED_THRESHOLD = 0.25


def load_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError(f"Cannot load fixed M1 config: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationBSupervisedError("Fixed M1 config must be an object")
    return value


def validate_config(path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(path)
    if actual_sha != EXPECTED_CONFIG_SHA256:
        raise EvaluationBSupervisedError(
            f"Evaluation B M1 config hash mismatch: expected {EXPECTED_CONFIG_SHA256}, got {actual_sha}"
        )
    config = load_config(path)
    if config.get("config_id") != "eval-b-m1-tfidf-logreg-v1" or config.get("method_id") != "M1":
        raise EvaluationBSupervisedError("Wrong Evaluation B M1 config identity")
    vectorizer = config["fixed_pipeline"]["vectorizer"]
    classifier = config["fixed_pipeline"]["base_classifier"]
    if (
        tuple(vectorizer.get("ngram_range", ())) != (1, 2)
        or int(vectorizer.get("min_df", -1)) != 2
        or float(classifier.get("C", -1)) != 1.0
        or classifier.get("class_weight") is not None
        or float(config["thresholding"].get("global_threshold", -1)) != FIXED_THRESHOLD
    ):
        raise EvaluationBSupervisedError("Evaluation B M1 fixed settings drifted")
    forbidden = config["scope_guard"]
    if any(
        bool(forbidden.get(key))
        for key in ("hyperparameter_search", "threshold_search", "feature_selection", "early_stopping", "human_labels_for_training_or_tuning", "evaluation_a_artifacts_mutable", "auxiliary_targets")
    ):
        raise EvaluationBSupervisedError("Evaluation B M1 scope guard permits a forbidden operation")
    source = config["selection_provenance"]
    source_path = REPO_ROOT / source["a1_run_metadata"]
    if sha256_file(source_path) != source["a1_run_metadata_sha256"]:
        raise EvaluationBSupervisedError("Frozen A1 M1 selection metadata hash mismatch")
    selected = json.loads(source_path.read_text(encoding="utf-8"))["selection"]
    if selected["selected_hyperparameters"] != {
        "vectorizer.min_df": 2,
        "base_classifier.C": 1.0,
        "base_classifier.class_weight": None,
    } or float(selected["selected_global_threshold"]) != FIXED_THRESHOLD:
        raise EvaluationBSupervisedError("Config does not exactly transfer the frozen A1 M1 selection")
    return config


def build_pipeline(config: Mapping[str, Any]) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    raw = config["fixed_pipeline"]
    v = raw["vectorizer"]
    c = raw["base_classifier"]
    vectorizer = TfidfVectorizer(
        analyzer=v["analyzer"],
        ngram_range=tuple(v["ngram_range"]),
        lowercase=bool(v["lowercase"]),
        strip_accents=v.get("strip_accents"),
        stop_words=v.get("stop_words"),
        token_pattern=v["token_pattern"],
        norm=v["norm"],
        use_idf=bool(v["use_idf"]),
        smooth_idf=bool(v["smooth_idf"]),
        sublinear_tf=bool(v["sublinear_tf"]),
        min_df=int(v["min_df"]),
        max_df=float(v["max_df"]),
        max_features=int(v["max_features"]),
        dtype=np.float64,
    )
    classifier = LogisticRegression(
        penalty=c["penalty"],
        solver=c["solver"],
        C=float(c["C"]),
        class_weight=c["class_weight"],
        max_iter=int(c["max_iter"]),
        tol=float(c["tol"]),
        random_state=int(c["random_state"]),
    )
    wrapper = OneVsRestClassifier(
        classifier, n_jobs=int(raw["classifier_wrapper"]["n_jobs"])
    )
    return vectorizer, wrapper


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".joblib", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _complete_is_valid(
    metadata_path: Path,
    *,
    model_path: Path,
    prediction_path: Path,
    config_sha256: str,
    membership_sha256: str,
    training_sha256: str,
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationBSupervisedError("Existing M1 metadata is unreadable") from exc
    if metadata.get("status") != "COMPLETE":
        raise EvaluationBSupervisedError(
            "Incomplete Evaluation B M1 artifacts exist; preserve them and choose a new output path"
        )
    expected = {
        "config_sha256": config_sha256,
        "retained_membership_sha256": membership_sha256,
        "training_membership_sha256": training_sha256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise EvaluationBSupervisedError("Existing M1 completion belongs to different frozen inputs")
    if not model_path.is_file() or sha256_file(model_path) != metadata.get("model_sha256"):
        raise EvaluationBSupervisedError("Existing Evaluation B M1 model artifact is damaged")
    if not prediction_path.is_file() or sha256_file(prediction_path) != metadata.get("prediction_sha256"):
        raise EvaluationBSupervisedError("Existing Evaluation B M1 prediction artifact is damaged")
    return True


def execute(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    prepared: Any,
    model_dir: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    config_sha = sha256_file(config_path)
    training_sha = training_membership_sha256(prepared.training_records)
    metadata_path = model_dir / "run_metadata.json"
    model_path = model_dir / "pipeline.joblib"
    if _complete_is_valid(
        metadata_path,
        model_path=model_path,
        prediction_path=prediction_path,
        config_sha256=config_sha,
        membership_sha256=prepared.membership_sha256,
        training_sha256=training_sha,
    ):
        return {"status": "SKIPPED_COMPLETE", "metadata_path": str(metadata_path)}
    unexpected = [path for path in (model_path, prediction_path) if path.exists()]
    if unexpected:
        raise EvaluationBSupervisedError(
            f"Partial Evaluation B M1 artifacts exist and will not be overwritten: {unexpected}"
        )

    lock = RunLock.acquire(model_dir)
    started = time.perf_counter()
    started_at = utc_now()
    run_id = os.urandom(12).hex()
    try:
        vectorizer, classifier = build_pipeline(config)
        x_train = vectorizer.fit_transform(texts(prepared.training_records))
        y_train = target_matrix(prepared.training_records, prepared.label_order)
        classifier.fit(x_train, y_train)
        target_texts = [case.fact_summary for case in prepared.target_cases]
        probabilities = np.asarray(classifier.predict_proba(vectorizer.transform(target_texts)), dtype=np.float64)
        rows = prediction_rows(
            method_id="M1",
            target_cases=prepared.target_cases,
            probabilities=probabilities,
            label_order=prepared.label_order,
            threshold=FIXED_THRESHOLD,
            run_id=run_id,
            config_sha256=config_sha,
            membership_sha256=prepared.membership_sha256,
            training_membership_sha256=training_sha,
        )
        _atomic_joblib(
            model_path,
            {
                "vectorizer": vectorizer,
                "classifier": classifier,
                "label_order": list(prepared.label_order),
                "threshold": FIXED_THRESHOLD,
            },
        )
        atomic_jsonl(prediction_path, rows)
        metadata = {
            "artifact_schema_version": "sherloc-eval-b-m1-run-v1",
            "runner_version": VERSION,
            "status": "COMPLETE",
            "run_id": run_id,
            "method_id": "M1",
            "evaluation": "B",
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha,
            "source_sha256": dict(prepared.source_hashes),
            "retained_membership_sha256": prepared.membership_sha256,
            "training_membership_sha256": training_sha,
            "source_silver_cohort_n": len(prepared.benchmark),
            "retained_n": prepared.retained_n,
            "retained_primary_cohort_n": prepared.retained_primary_cohort_n,
            "train_n": prepared.train_n,
            "prediction_n": len(rows),
            "training_label_supports": dict(prepared.training_label_supports),
            "label_order": list(prepared.label_order),
            "fixed_hyperparameters": {
                "tfidf_ngram_range": [1, 2],
                "min_df": 2,
                "C": 1.0,
                "class_weight": None,
                "global_threshold": FIXED_THRESHOLD,
            },
            "selection_policy": "TRANSFERRED_A1_SETTINGS_NO_EVALUATION_B_SELECTION",
            "human_labels_used_for_training_tuning_or_prediction": False,
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": sha256_file(prediction_path),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        }
        atomic_json(metadata_path, metadata)
        return {
            "status": "COMPLETE",
            "train_n": prepared.train_n,
            "prediction_n": len(rows),
            "metadata_path": str(metadata_path),
        }
    finally:
        lock.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="Write exclusion audit; never fit")
    mode.add_argument("--execute", action="store_true", help="Fit and predict after explicit freeze confirmation")
    parser.add_argument("--confirm-qc-membership-freeze", action="store_true")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--reliability-sample", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    parser.add_argument("--human-reference", type=Path, default=DEFAULT_HUMAN_REFERENCE)
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument("--membership-freeze", type=Path, default=DEFAULT_MEMBERSHIP_FREEZE)
    parser.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    parser.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prediction-output", type=Path, default=DEFAULT_PREDICTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = validate_config(args.config)
    if sha256_file(DEFAULT_M2_CONFIG) != EXPECTED_M2_CONFIG_SHA256:
        raise EvaluationBSupervisedError("Companion Evaluation B M2 config hash mismatch")
    prepared = prepare_supervised_data(
        benchmark_path=args.benchmark,
        ontology_path=args.ontology,
        reliability_sample_path=args.reliability_sample,
        human_reference_path=args.human_reference,
        membership_path=args.membership,
        membership_freeze_path=args.membership_freeze,
        a1_split_path=args.a1_split,
        a2_split_path=args.a2_split,
    )
    preflight = write_preflight_artifacts(
        prepared,
        audit_path=args.audit_output,
        preflight_path=args.preflight_output,
        config_hashes={
            "M1": sha256_file(args.config),
            "M2": sha256_file(DEFAULT_M2_CONFIG),
        },
    )
    if args.preflight:
        print(canonical_json({"mode": "PREFLIGHT", **preflight}))
        return 0
    if not args.confirm_qc_membership_freeze:
        raise EvaluationBSupervisedError(
            "Execution requires --confirm-qc-membership-freeze after root confirms the freeze"
        )
    result = execute(
        config=config,
        config_path=args.config,
        prepared=prepared,
        model_dir=args.model_dir,
        prediction_path=args.prediction_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except EvaluationBSupervisedError as exc:
        print(f"Evaluation B M1 protocol error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
