#!/usr/bin/env python3
"""Evaluate the frozen 55-case human-grounded auxiliary LLM extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "src/experiments/23_run_evaluation_b_auxiliary_llm.py"
DEFAULT_REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
DEFAULT_PREDICTIONS = (
    REPO_ROOT / "outputs/predictions/evaluation_b/auxiliary_zero_shot/predictions.jsonl"
)
DEFAULT_DIAGNOSTICS = REPO_ROOT / "outputs/logs/evaluation_b/auxiliary_llm/diagnostics.json"
DEFAULT_RESULTS = (
    REPO_ROOT / "outputs/analysis/evaluation_b/auxiliary_llm_human_grounded_results.csv"
)
DEFAULT_PER_CLASS = (
    REPO_ROOT / "outputs/analysis/evaluation_b/auxiliary_llm_per_class_results.csv"
)
DEFAULT_CASE_LEVEL = (
    REPO_ROOT / "outputs/analysis/evaluation_b/auxiliary_llm_case_level.csv"
)
DEFAULT_COMPLETION = (
    REPO_ROOT / "outputs/analysis/evaluation_b/auxiliary_llm_completion_manifest.json"
)
GEO = ("Internal", "Transnational")
MULTIPLICITY = ("SINGLE", "MULTIPLE", "UNKNOWN")
CHILD = ("TRUE", "FALSE", "UNKNOWN")
OCG = ("TRUE", "FALSE")


class AuxiliaryEvaluationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuxiliaryEvaluationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuxiliaryEvaluationError(f"Expected a JSON object at {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise AuxiliaryEvaluationError(f"Cannot read {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        raise AuxiliaryEvaluationError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def load_runner() -> ModuleType:
    name = "_eval_b_auxiliary_runner_for_evaluation"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise AuxiliaryEvaluationError("Cannot import auxiliary runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bool(value: Any, *, field: str) -> bool:
    normalized = str(value or "").strip().upper()
    if normalized in {"1", "TRUE"}:
        return True
    if normalized in {"0", "FALSE"}:
        return False
    raise AuxiliaryEvaluationError(f"Invalid boolean {field}: {value!r}")


def _geo(value: Any, *, field: str) -> tuple[str, ...]:
    try:
        raw = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise AuxiliaryEvaluationError(f"Invalid JSON in {field}") from exc
    if (
        not isinstance(raw, list)
        or any(item not in GEO for item in raw)
        or len(raw) != len(set(raw))
    ):
        raise AuxiliaryEvaluationError(f"Invalid Geographic Form in {field}")
    selected = set(raw)
    return tuple(item for item in GEO if item in selected)


def load_reference(path: Path = DEFAULT_REFERENCE) -> list[dict[str, Any]]:
    rows = load_csv(path)
    substantive = [
        row for row in rows if str(row.get("review_status") or "").strip() == "SUBSTANTIVE"
    ]
    if len(rows) != 61 or len(substantive) != 55:
        raise AuxiliaryEvaluationError("Frozen reference must contain 61 retained / 55 substantive")
    output: list[dict[str, Any]] = []
    for row in substantive:
        case_id = str(row["reliability_case_id"])
        auxiliary_mask = _bool(row["auxiliary_evaluable"], field=f"{case_id}.auxiliary")
        ocg_mask = _bool(
            row["organized_criminal_group_evaluable"], field=f"{case_id}.ocg_evaluable"
        )
        multiplicity = str(row["multiplicity_human_clean"]).strip()
        child = str(row["child_human_clean"]).strip()
        geo = _geo(row["geographic_form_human_clean_json"], field=f"{case_id}.geo")
        ocg = str(row["organized_criminal_group_human"]).strip()
        output.append(
            {
                "reliability_case_id": case_id,
                "search_rank": int(row["search_rank"]),
                "canonical_url": str(row["canonical_url"]),
                "jurisdiction": str(row["jurisdiction"]),
                "fact_summary": str(row["fact_summary"]),
                "input_sha256": str(row["input_sha256"]),
                "geo": geo,
                "geo_evaluable": auxiliary_mask,
                "multiplicity": multiplicity,
                "multiplicity_evaluable": auxiliary_mask
                and multiplicity in MULTIPLICITY
                and multiplicity != "Not Applicable",
                "child": child,
                "child_evaluable": auxiliary_mask
                and child in CHILD
                and child != "Not Applicable",
                "ocg": ocg,
                "ocg_evaluable": ocg_mask and ocg in OCG,
            }
        )
    output.sort(key=lambda row: int(row["search_rank"]))
    return output


def validate_predictions(
    runner: ModuleType,
    references: Sequence[Mapping[str, Any]],
    prediction_path: Path = DEFAULT_PREDICTIONS,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
) -> dict[str, dict[str, Any]]:
    predictions = load_jsonl(prediction_path)
    diagnostics = load_json(diagnostics_path)
    prepared = runner.prepare()
    reference_ids = {str(row["reliability_case_id"]) for row in references}
    observed_ids = {str(row.get("reliability_case_id") or "") for row in predictions}
    if (
        len(predictions) != 55
        or len(observed_ids) != 55
        or observed_ids != reference_ids
        or diagnostics.get("status") != "COMPLETE"
        or diagnostics.get("successful_predictions") != 55
        or diagnostics.get("unresolved_failures") != 0
        or diagnostics.get("missing_unattempted") != 0
        or diagnostics.get("prediction_sha256") != sha256_file(prediction_path)
        or diagnostics.get("membership_sha256") != prepared["membership_sha256"]
        or diagnostics.get("config_sha256") != prepared["contract"]["config_sha256"]
        or diagnostics.get("prompt_sha256") != prepared["contract"]["prompt_sha256"]
        or diagnostics.get("schema_sha256") != prepared["contract"]["schema_sha256"]
        or diagnostics.get("zero_shot") is not True
        or diagnostics.get("store") is not False
        or diagnostics.get("human_or_silver_labels_sent_to_model") is not False
    ):
        raise AuxiliaryEvaluationError("Auxiliary prediction/diagnostic completion gate failed")
    reference_by_id = {str(row["reliability_case_id"]): row for row in references}
    result: dict[str, dict[str, Any]] = {}
    for row in predictions:
        case_id = str(row["reliability_case_id"])
        reference = reference_by_id[case_id]
        expected = {
            "status": "SUCCESS_VALIDATED",
            "method": "AUX_LLM_ZERO_SHOT",
            "search_rank": int(reference["search_rank"]),
            "canonical_url": reference["canonical_url"],
            "jurisdiction": reference["jurisdiction"],
            "fact_summary": reference["fact_summary"],
            "input_sha256": reference["input_sha256"],
            "membership_sha256": prepared["membership_sha256"],
            "prompt_sha256": prepared["contract"]["prompt_sha256"],
            "schema_sha256": prepared["contract"]["schema_sha256"],
            "config_sha256": prepared["contract"]["config_sha256"],
            "demonstration_count": 0,
            "store": False,
            "human_or_silver_labels_sent_to_model": False,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise AuxiliaryEvaluationError(f"Prediction provenance mismatch for {case_id}")
        prediction = runner.validate_output(row.get("validated_prediction"))
        if row.get("validated_prediction") != prediction:
            raise AuxiliaryEvaluationError(f"Prediction is not canonical for {case_id}")
        result[case_id] = dict(row)
    return result


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_counts(reference: Sequence[bool], prediction: Sequence[bool]) -> dict[str, int]:
    if len(reference) != len(prediction):
        raise AuxiliaryEvaluationError("Metric vectors differ in length")
    return {
        "tp": sum(left and right for left, right in zip(reference, prediction, strict=True)),
        "fp": sum(not left and right for left, right in zip(reference, prediction, strict=True)),
        "fn": sum(left and not right for left, right in zip(reference, prediction, strict=True)),
        "tn": sum(not left and not right for left, right in zip(reference, prediction, strict=True)),
    }


def _binary_metrics(counts: Mapping[str, int]) -> dict[str, float]:
    precision = _divide(counts["tp"], counts["tp"] + counts["fp"])
    recall = _divide(counts["tp"], counts["tp"] + counts["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
    }


def _format(value: float) -> str:
    if not math.isfinite(value):
        raise AuxiliaryEvaluationError("Metric is not finite")
    return f"{value:.12f}"


def _aggregate_row(
    target: str,
    metric: str,
    value: float,
    n: int,
    support: Mapping[str, int],
    confusion: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    return {
        "target": target,
        "metric": metric,
        "value": _format(value),
        "n": n,
        "support_json": canonical_json(support),
        "confusion_matrix_json": canonical_json(confusion) if confusion is not None else "",
        "model": "gpt-5.6-luna",
        "prompt_version": "eval-b-auxiliary-zero-shot-v1",
    }


def _per_class_rows(
    target: str,
    classes: Sequence[str],
    reference: Sequence[str],
    prediction: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in classes:
        counts = _binary_counts(
            [value == label for value in reference], [value == label for value in prediction]
        )
        metrics = _binary_metrics(counts)
        rows.append(
            {
                "target": target,
                "class": label,
                "n": len(reference),
                "support": counts["tp"] + counts["fn"],
                **counts,
                "precision": _format(metrics["precision"]),
                "recall": _format(metrics["recall"]),
                "f1": _format(metrics["f1"]),
            }
        )
    return rows


def evaluate(
    references: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    aggregates: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    geo_reference: list[set[str]] = []
    geo_prediction: list[set[str]] = []
    categorical: dict[str, tuple[tuple[str, ...], list[str], list[str]]] = {
        "VICTIM_MULTIPLICITY": (MULTIPLICITY, [], []),
        "CHILD_INVOLVEMENT": (CHILD, [], []),
        "ORGANIZED_CRIMINAL_GROUP": (OCG, [], []),
    }
    for human in references:
        case_id = str(human["reliability_case_id"])
        record = predictions[case_id]
        predicted = record["validated_prediction"]
        geo_human = tuple(human["geo"])
        geo_predicted = tuple(predicted["geographic_form"])
        geo_evaluable = bool(human["geo_evaluable"])
        multiplicity_evaluable = bool(human["multiplicity_evaluable"])
        child_evaluable = bool(human["child_evaluable"])
        ocg_evaluable = bool(human["ocg_evaluable"])
        if geo_evaluable:
            geo_reference.append(set(geo_human))
            geo_prediction.append(set(geo_predicted))
        if multiplicity_evaluable:
            categorical["VICTIM_MULTIPLICITY"][1].append(str(human["multiplicity"]))
            categorical["VICTIM_MULTIPLICITY"][2].append(str(predicted["multiplicity"]))
        if child_evaluable:
            categorical["CHILD_INVOLVEMENT"][1].append(str(human["child"]))
            categorical["CHILD_INVOLVEMENT"][2].append(str(predicted["child_involvement"]))
        if ocg_evaluable:
            categorical["ORGANIZED_CRIMINAL_GROUP"][1].append(str(human["ocg"]))
            categorical["ORGANIZED_CRIMINAL_GROUP"][2].append(
                str(predicted["organized_criminal_group"])
            )
        case_rows.append(
            {
                "reliability_case_id": case_id,
                "search_rank": int(human["search_rank"]),
                "canonical_url": human["canonical_url"],
                "jurisdiction": human["jurisdiction"],
                "fact_summary": human["fact_summary"],
                "input_sha256": human["input_sha256"],
                "prediction_request_sha256": record["request_sha256"],
                "geographic_form_human_json": canonical_json(list(geo_human)),
                "geographic_form_prediction_json": canonical_json(list(geo_predicted)),
                "geographic_form_evaluable": int(geo_evaluable),
                "geographic_form_correct": (
                    int(set(geo_human) == set(geo_predicted)) if geo_evaluable else ""
                ),
                "multiplicity_human": human["multiplicity"],
                "multiplicity_prediction": predicted["multiplicity"],
                "multiplicity_evaluable": int(multiplicity_evaluable),
                "multiplicity_correct": (
                    int(human["multiplicity"] == predicted["multiplicity"])
                    if multiplicity_evaluable
                    else ""
                ),
                "child_involvement_human": human["child"],
                "child_involvement_prediction": predicted["child_involvement"],
                "child_involvement_evaluable": int(child_evaluable),
                "child_involvement_correct": (
                    int(human["child"] == predicted["child_involvement"])
                    if child_evaluable
                    else ""
                ),
                "organized_criminal_group_human": human["ocg"],
                "organized_criminal_group_prediction": predicted[
                    "organized_criminal_group"
                ],
                "organized_criminal_group_evaluable": int(ocg_evaluable),
                "organized_criminal_group_correct": (
                    int(human["ocg"] == predicted["organized_criminal_group"])
                    if ocg_evaluable
                    else ""
                ),
            }
        )

    geo_class_rows: list[dict[str, Any]] = []
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for label in GEO:
        counts = _binary_counts(
            [label in value for value in geo_reference],
            [label in value for value in geo_prediction],
        )
        metrics = _binary_metrics(counts)
        for key in total_counts:
            total_counts[key] += counts[key]
        geo_class_rows.append(
            {
                "target": "GEOGRAPHIC_FORM",
                "class": label,
                "n": len(geo_reference),
                "support": counts["tp"] + counts["fn"],
                **counts,
                "precision": _format(metrics["precision"]),
                "recall": _format(metrics["recall"]),
                "f1": _format(metrics["f1"]),
            }
        )
    per_class.extend(geo_class_rows)
    geo_support = {row["class"]: int(row["support"]) for row in geo_class_rows}
    macro = sum(float(row["f1"]) for row in geo_class_rows) / len(GEO)
    micro = _binary_metrics(total_counts)["f1"]
    exact = sum(left == right for left, right in zip(geo_reference, geo_prediction, strict=True))
    jaccard = sum(
        len(left & right) / len(left | right) if left | right else 1.0
        for left, right in zip(geo_reference, geo_prediction, strict=True)
    )
    for metric, value in (
        ("MACRO_F1", macro),
        ("MICRO_F1", micro),
        ("EXACT_SET_ACCURACY", _divide(exact, len(geo_reference))),
        ("EXAMPLE_JACCARD", _divide(jaccard, len(geo_reference))),
    ):
        aggregates.append(
            _aggregate_row("GEOGRAPHIC_FORM", metric, value, len(geo_reference), geo_support)
        )

    for target, (classes, reference, prediction) in categorical.items():
        class_rows = _per_class_rows(target, classes, reference, prediction)
        per_class.extend(class_rows)
        support = {row["class"]: int(row["support"]) for row in class_rows}
        confusion = {
            truth: {pred: 0 for pred in classes}
            for truth in classes
        }
        for truth, pred in zip(reference, prediction, strict=True):
            confusion[truth][pred] += 1
        accuracy = _divide(
            sum(left == right for left, right in zip(reference, prediction, strict=True)),
            len(reference),
        )
        if target == "ORGANIZED_CRIMINAL_GROUP":
            positive = next(row for row in class_rows if row["class"] == "TRUE")
            true_metrics = {
                key: float(positive[key]) for key in ("precision", "recall", "f1")
            }
            false_recall = float(next(row for row in class_rows if row["class"] == "FALSE")["recall"])
            metric_values = (
                ("ACCURACY", accuracy),
                ("PRECISION", true_metrics["precision"]),
                ("RECALL", true_metrics["recall"]),
                ("F1", true_metrics["f1"]),
                ("BALANCED_ACCURACY", (true_metrics["recall"] + false_recall) / 2),
            )
        else:
            metric_values = (
                ("ACCURACY", accuracy),
                ("MACRO_F1", sum(float(row["f1"]) for row in class_rows) / len(classes)),
            )
        for metric, value in metric_values:
            aggregates.append(
                _aggregate_row(target, metric, value, len(reference), support, confusion)
            )
    return aggregates, per_class, case_rows


AGGREGATE_FIELDS = (
    "target",
    "metric",
    "value",
    "n",
    "support_json",
    "confusion_matrix_json",
    "model",
    "prompt_version",
)
PER_CLASS_FIELDS = (
    "target",
    "class",
    "n",
    "support",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
)
CASE_FIELDS = (
    "reliability_case_id",
    "search_rank",
    "canonical_url",
    "jurisdiction",
    "fact_summary",
    "input_sha256",
    "prediction_request_sha256",
    "geographic_form_human_json",
    "geographic_form_prediction_json",
    "geographic_form_evaluable",
    "geographic_form_correct",
    "multiplicity_human",
    "multiplicity_prediction",
    "multiplicity_evaluable",
    "multiplicity_correct",
    "child_involvement_human",
    "child_involvement_prediction",
    "child_involvement_evaluable",
    "child_involvement_correct",
    "organized_criminal_group_human",
    "organized_criminal_group_prediction",
    "organized_criminal_group_evaluable",
    "organized_criminal_group_correct",
)


def run(
    *,
    reference_path: Path = DEFAULT_REFERENCE,
    prediction_path: Path = DEFAULT_PREDICTIONS,
    diagnostics_path: Path = DEFAULT_DIAGNOSTICS,
    results_path: Path = DEFAULT_RESULTS,
    per_class_path: Path = DEFAULT_PER_CLASS,
    case_level_path: Path = DEFAULT_CASE_LEVEL,
    completion_path: Path = DEFAULT_COMPLETION,
) -> dict[str, Any]:
    runner = load_runner()
    references = load_reference(reference_path)
    predictions = validate_predictions(runner, references, prediction_path, diagnostics_path)
    prepared = runner.prepare()
    config = prepared["contract"]["config"]
    freeze_path = REPO_ROOT / config["outputs"]["pre_execution_freeze"]
    if not freeze_path.is_file():
        raise AuxiliaryEvaluationError("Pre-execution freeze artifact is missing")
    observed_freeze = load_json(freeze_path)
    current_freeze = runner.pre_execution_freeze(prepared)
    observed_comparable = dict(observed_freeze)
    current_comparable = dict(current_freeze)
    observed_comparable.pop("frozen_at", None)
    current_comparable.pop("frozen_at", None)
    if observed_comparable != current_comparable:
        raise AuxiliaryEvaluationError("Pre-execution freeze no longer matches the inputs")
    aggregates, per_class, cases = evaluate(references, predictions)
    write_csv(results_path, aggregates, AGGREGATE_FIELDS)
    write_csv(per_class_path, per_class, PER_CLASS_FIELDS)
    write_csv(case_level_path, cases, CASE_FIELDS)
    artifacts = {
        "config": prepared["contract"]["config_path"],
        "prompt": prepared["contract"]["prompt_path"],
        "schema": prepared["contract"]["schema_path"],
        "human_reference": REPO_ROOT / config["membership"]["reference_path"],
        "evaluation_b_membership_manifest": (
            REPO_ROOT / config["membership"]["membership_manifest_path"]
        ),
        "pre_execution_freeze": freeze_path,
        "predictions": prediction_path,
        "diagnostics": diagnostics_path,
        "aggregate_metrics": results_path,
        "per_class_metrics": per_class_path,
        "case_level": case_level_path,
        "runner": RUNNER_PATH,
        "evaluator": Path(__file__),
    }
    nested_artifacts = {
        name: {
            "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": sha256_file(path),
        }
        for name, path in artifacts.items()
    }
    artifacts_sha256 = {
        metadata["path"]: metadata["sha256"]
        for metadata in nested_artifacts.values()
    }
    manifest = {
        "schema_version": "sherloc-eval-b-auxiliary-completion-v1",
        "status": "COMPLETE",
        "evaluation": "A3_HUMAN_GROUNDED_AUXILIARY",
        "model": "gpt-5.6-luna",
        "zero_shot": True,
        "substantive_membership_n": 55,
        "target_n": {
            target: int(next(row["n"] for row in aggregates if row["target"] == target))
            for target in (
                "GEOGRAPHIC_FORM",
                "VICTIM_MULTIPLICITY",
                "CHILD_INVOLVEMENT",
                "ORGANIZED_CRIMINAL_GROUP",
            )
        },
        "store": False,
        "human_or_silver_labels_sent_to_model": False,
        "human_reference_sha256": sha256_file(reference_path),
        "evaluation_b_membership_manifest_sha256": sha256_file(
            REPO_ROOT / config["membership"]["membership_manifest_path"]
        ),
        "artifacts": nested_artifacts,
        "artifacts_sha256": artifacts_sha256,
    }
    write_json(completion_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    args = parser.parse_args(argv)
    try:
        manifest = run(prediction_path=args.predictions)
    except AuxiliaryEvaluationError as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
