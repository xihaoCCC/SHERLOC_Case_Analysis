#!/usr/bin/env python3
"""Generate paired-bootstrap method-difference intervals for the final paper.

This is a read-only analysis of the frozen A1, A2, and Evaluation-B AMP
predictions.  It neither trains a model nor calls an API.  The only file it
writes is the requested paper-final paired-difference table.

For every evaluation, one deterministic case-index matrix is generated and
used for M2, M3, and M4.  Consequently, both methods in every contrast are
evaluated on the exact same resampled cases in every replicate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.metrics import (  # noqa: E402
    AMP_LABEL_IDS,
    ORGAN_REMOVAL_LABEL,
    compute_amp_cpmr,
    compute_amp_metrics,
    labels_to_indicator,
    supported_label_ids,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/analysis/paper_final/paired_bootstrap_method_differences.csv"
)
DEFAULT_EVAL_A_MANIFEST = REPO_ROOT / "outputs/metrics/amp_evaluation_manifest.json"
DEFAULT_EVAL_B_MANIFEST = (
    REPO_ROOT / "outputs/analysis/evaluation_b/evaluation_b_analysis_manifest.json"
)
DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20260811
CONFIDENCE_LEVEL = 0.95

METHODS = ("M2", "M3", "M4")
COMPARISONS = (("M3", "M2"), ("M4", "M2"), ("M4", "M3"))
EXPECTED_N = {"A1": 253, "A2": 861, "A3": 55}
REFERENCE_TYPE = {
    "A1": "SHERLOC silver reference",
    "A2": "SHERLOC silver reference",
    "A3": "human-grounded narrative reference",
}
METRIC_DISPLAY = {
    "macro_f1": "Macro-F1",
    "micro_f1": "Micro-F1",
    "exact_set_accuracy": "Exact-set accuracy",
    "example_jaccard": "Example-based Jaccard",
    "act_cpmr": "Act CPMR",
    "means_cpmr": "Means CPMR",
    "purpose_cpmr": "Purpose CPMR",
}

OUTPUT_FIELDS = (
    "evaluation",
    "reference_type",
    "comparison",
    "first_method",
    "second_method",
    "metric",
    "n",
    "point_difference",
    "ci_low",
    "ci_high",
    "confidence_level",
    "bootstrap_resamples",
    "seed",
    "resampling_unit",
    "bootstrap_method",
    "macro_label_count",
    "macro_label_ids_json",
    "ci_excludes_zero",
)


class PairedBootstrapError(RuntimeError):
    """Raised when a frozen input or paired-analysis invariant is violated."""


@dataclass(frozen=True)
class EvaluationMatrices:
    evaluation: str
    reference: np.ndarray
    predictions: Mapping[str, np.ndarray]
    macro_label_ids: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.reference.shape[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PairedBootstrapError(f"Required JSON does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairedBootstrapError(f"Malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PairedBootstrapError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise PairedBootstrapError(f"Required CSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PairedBootstrapError(f"CSV has no header: {path}")
        return list(reader)


def _resolve_manifest_path(raw: Any, *, artifact: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise PairedBootstrapError(f"{artifact} has a blank path")
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise PairedBootstrapError(
            f"{artifact} resolves outside the repository: {resolved}"
        ) from exc
    return resolved


def _validate_hash(path: Path, expected: Any, *, artifact: str) -> None:
    digest = str(expected or "").strip()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise PairedBootstrapError(f"{artifact} has a malformed frozen SHA-256")
    if not path.is_file():
        raise PairedBootstrapError(f"Frozen {artifact} is missing: {path}")
    observed = _sha256_file(path)
    if observed != digest:
        raise PairedBootstrapError(
            f"Frozen {artifact} hash mismatch: expected {digest}, observed {observed}"
        )


def _validate_eval_b_manifest_hashes(manifest: Mapping[str, Any]) -> None:
    """Verify every canonical Evaluation-B input/output bound by its manifest."""

    for section in ("inputs_sha256", "outputs_sha256"):
        entries = manifest.get(section)
        if not isinstance(entries, Mapping) or not entries:
            raise PairedBootstrapError(f"Evaluation B manifest lacks {section}")
        for relative, digest in entries.items():
            path = _resolve_manifest_path(relative, artifact=f"Evaluation B {section}")
            _validate_hash(path, digest, artifact=f"Evaluation B {section} {relative}")


def _point_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
    macro_label_ids: Sequence[str],
) -> dict[str, float]:
    aggregate = compute_amp_metrics(
        reference,
        prediction,
        macro_label_ids=tuple(macro_label_ids),
    )
    cpmr = compute_amp_cpmr(reference, prediction)["by_family"]
    return {
        "macro_f1": float(aggregate["macro_f1"]),
        "micro_f1": float(aggregate["micro_f1"]),
        "exact_set_accuracy": float(aggregate["exact_set_accuracy"]),
        "example_jaccard": float(aggregate["example_jaccard"]),
        "act_cpmr": float(cpmr["ACT"]["cpmr"]),
        "means_cpmr": float(cpmr["MEANS"]["cpmr"]),
        "purpose_cpmr": float(cpmr["PURPOSE"]["cpmr"]),
    }


def paired_bootstrap_rows(
    data: EvaluationMatrices,
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Compute paired case-bootstrap differences for the three frozen contrasts."""

    if data.evaluation not in EXPECTED_N:
        raise PairedBootstrapError(f"Unknown evaluation: {data.evaluation}")
    if data.n != EXPECTED_N[data.evaluation]:
        raise PairedBootstrapError(
            f"{data.evaluation} N is {data.n}; expected {EXPECTED_N[data.evaluation]}"
        )
    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise PairedBootstrapError("bootstrap resamples must be a positive integer")
    if not isinstance(seed, int):
        raise PairedBootstrapError("bootstrap seed must be an integer")
    if set(data.predictions) != set(METHODS):
        raise PairedBootstrapError(
            f"{data.evaluation} predictions must contain exactly {list(METHODS)}"
        )
    for method, matrix in data.predictions.items():
        if matrix.shape != data.reference.shape:
            raise PairedBootstrapError(
                f"{data.evaluation} {method} matrix shape differs from reference"
            )

    point = {
        method: _point_metrics(data.reference, data.predictions[method], data.macro_label_ids)
        for method in METHODS
    }
    samples = {
        method: {
            metric: np.empty(n_resamples, dtype=np.float64) for metric in METRIC_DISPLAY
        }
        for method in METHODS
    }

    # One sampled-index matrix per evaluation.  Every method and therefore
    # every contrast uses the identical indices at every replicate.
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(0, data.n, size=(n_resamples, data.n))
    for sample_number, indices in enumerate(sampled_indices):
        sampled_reference = data.reference[indices]
        for method in METHODS:
            values = _point_metrics(
                sampled_reference,
                data.predictions[method][indices],
                data.macro_label_ids,
            )
            for metric, value in values.items():
                samples[method][metric][sample_number] = value

    rows: list[dict[str, Any]] = []
    macro_json = json.dumps(list(data.macro_label_ids), separators=(",", ":"))
    for first_method, second_method in COMPARISONS:
        for metric, display_name in METRIC_DISPLAY.items():
            differences = samples[first_method][metric] - samples[second_method][metric]
            ci_low, ci_high = np.percentile(
                differences, (2.5, 97.5), method="linear"
            )
            point_difference = point[first_method][metric] - point[second_method][metric]
            rows.append(
                {
                    "evaluation": data.evaluation,
                    "reference_type": REFERENCE_TYPE[data.evaluation],
                    "comparison": f"{first_method} - {second_method}",
                    "first_method": first_method,
                    "second_method": second_method,
                    "metric": display_name,
                    "n": data.n,
                    "point_difference": float(point_difference),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "confidence_level": CONFIDENCE_LEVEL,
                    "bootstrap_resamples": n_resamples,
                    "seed": seed,
                    "resampling_unit": "CASE_WITH_ALL_17_LABELS",
                    "bootstrap_method": "PAIRED_CASE_RESAMPLING_PERCENTILE_LINEAR",
                    "macro_label_count": len(data.macro_label_ids),
                    "macro_label_ids_json": macro_json,
                    "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                }
            )
    return rows


def _canonical_metric_rows(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip().upper()
        variant = str(row.get("prediction_variant", "PRIMARY")).strip().upper()
        if method in METHODS and variant == "PRIMARY":
            if method in output:
                raise PairedBootstrapError(f"Duplicate {method} canonical row in {path}")
            output[method] = row
    if set(output) != set(METHODS):
        raise PairedBootstrapError(f"Canonical table {path} lacks M2/M3/M4 PRIMARY rows")
    return output


def _assert_close(actual: float, expected: Any, *, field: str) -> None:
    try:
        target = float(expected)
    except (TypeError, ValueError) as exc:
        raise PairedBootstrapError(f"Canonical {field} is not numeric: {expected!r}") from exc
    if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-12):
        raise PairedBootstrapError(
            f"Recomputed {field} ({actual}) differs from canonical value ({target})"
        )


def _validate_point_metrics(
    data: EvaluationMatrices,
    canonical_path: Path,
) -> None:
    canonical = _canonical_metric_rows(canonical_path)
    field_by_metric = {
        "A1": {
            "macro_f1": "macro_f1",
            "micro_f1": "micro_f1",
            "exact_set_accuracy": "exact_set_accuracy",
            "example_jaccard": "example_jaccard",
            "act_cpmr": "act_cpmr",
            "means_cpmr": "means_cpmr",
            "purpose_cpmr": "purpose_cpmr",
        },
        "A2": {
            "macro_f1": "pooled_ood_macro_f1",
            "micro_f1": "pooled_micro_f1",
            "exact_set_accuracy": "pooled_exact_set_accuracy",
            "example_jaccard": "pooled_example_jaccard",
            "act_cpmr": "pooled_act_cpmr",
            "means_cpmr": "pooled_means_cpmr",
            "purpose_cpmr": "pooled_purpose_cpmr",
        },
        "A3": {
            "macro_f1": "macro_f1",
            "micro_f1": "micro_f1",
            "exact_set_accuracy": "exact_set",
            "example_jaccard": "jaccard",
            "act_cpmr": "act_cpmr",
            "means_cpmr": "means_cpmr",
            "purpose_cpmr": "purpose_cpmr",
        },
    }[data.evaluation]
    for method in METHODS:
        values = _point_metrics(data.reference, data.predictions[method], data.macro_label_ids)
        for metric, value in values.items():
            canonical_field = field_by_metric[metric]
            _assert_close(
                value,
                canonical[method].get(canonical_field),
                field=f"{data.evaluation}.{method}.{canonical_field}",
            )


def load_evaluation_a(
    manifest_path: Path = DEFAULT_EVAL_A_MANIFEST,
) -> dict[str, EvaluationMatrices]:
    """Load A1/A2 and fail closed against final splits and the canonical manifest."""

    evaluator = importlib.import_module("src.experiments.11_evaluate_amp")
    manifest = _read_json(manifest_path)
    if manifest.get("final_completion_gate") != "PASSED_M1_M2_M3_M4_A1_A2":
        raise PairedBootstrapError("Evaluation A final completion gate is not passed")
    bootstrap = manifest.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping) or (
        int(bootstrap.get("n_resamples", -1)) != DEFAULT_BOOTSTRAP_RESAMPLES
        or int(bootstrap.get("seed", -1)) != DEFAULT_BOOTSTRAP_SEED
    ):
        raise PairedBootstrapError("Evaluation A canonical bootstrap policy has drifted")

    input_entries = manifest.get("input_files")
    if not isinstance(input_entries, list) or len(input_entries) != 16:
        raise PairedBootstrapError("Evaluation A manifest must bind exactly 16 prediction files")
    prediction_paths: list[Path] = []
    observed_relative_paths: set[str] = set()
    for index, entry in enumerate(input_entries):
        if not isinstance(entry, Mapping):
            raise PairedBootstrapError(f"Malformed Evaluation A input_files[{index}]")
        path = _resolve_manifest_path(entry.get("path"), artifact="Evaluation A prediction")
        _validate_hash(path, entry.get("sha256"), artifact=f"Evaluation A prediction {path.name}")
        relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
        if not relative.startswith("outputs/predictions/"):
            raise PairedBootstrapError(f"Unexpected Evaluation A prediction path: {relative}")
        if relative in observed_relative_paths:
            raise PairedBootstrapError(f"Duplicate Evaluation A prediction binding: {relative}")
        observed_relative_paths.add(relative)
        prediction_paths.append(path)

    records = evaluator.load_prediction_files(prediction_paths)
    evaluator.validate_common_test_membership(records)
    split_diagnostics = evaluator.validate_against_final_splits(records)
    for evaluation, expected_n in (("A1", 253), ("A2", 861)):
        prefix = evaluation.lower()
        if split_diagnostics.get(f"{prefix}_final_split_validated") is not True:
            raise PairedBootstrapError(f"{evaluation} final split was not validated")
        if int(split_diagnostics.get(f"{prefix}_expected_test_n", -1)) != expected_n:
            raise PairedBootstrapError(f"{evaluation} final split N has drifted")

    output: dict[str, EvaluationMatrices] = {}
    for evaluation in ("A1", "A2"):
        by_method: dict[str, dict[tuple[int, int], Any]] = {}
        for method in METHODS:
            subset = [
                row
                for row in records
                if row.evaluation == evaluation
                and row.method == method
                and row.prediction_variant == evaluator.PRIMARY_VARIANT
            ]
            keyed = {(int(row.fold or 0), row.search_rank): row for row in subset}
            if len(keyed) != len(subset):
                raise PairedBootstrapError(f"Duplicate {method} {evaluation} case identity")
            by_method[method] = keyed
        expected_keys = set(by_method["M2"])
        if len(expected_keys) != EXPECTED_N[evaluation]:
            raise PairedBootstrapError(
                f"{evaluation} M2 membership N={len(expected_keys)}; expected {EXPECTED_N[evaluation]}"
            )
        for method in METHODS[1:]:
            if set(by_method[method]) != expected_keys:
                raise PairedBootstrapError(f"{evaluation} {method}/M2 membership mismatch")

        ordered_keys = sorted(expected_keys)
        reference_labels: list[tuple[str, ...]] = []
        prediction_labels: dict[str, list[tuple[str, ...]]] = {method: [] for method in METHODS}
        for key in ordered_keys:
            anchor = by_method["M2"][key]
            reference_labels.append(anchor.silver_reference_labels)
            for method in METHODS:
                row = by_method[method][key]
                identity = (
                    row.search_rank,
                    row.fold,
                    row.case_id,
                    row.canonical_url,
                    row.jurisdiction,
                    row.fact_summary,
                    row.silver_reference_labels,
                )
                anchor_identity = (
                    anchor.search_rank,
                    anchor.fold,
                    anchor.case_id,
                    anchor.canonical_url,
                    anchor.jurisdiction,
                    anchor.fact_summary,
                    anchor.silver_reference_labels,
                )
                if identity != anchor_identity:
                    raise PairedBootstrapError(
                        f"{evaluation} provenance/reference mismatch for {method} at rank {anchor.search_rank}"
                    )
                prediction_labels[method].append(row.predicted_labels)

        reference = labels_to_indicator(reference_labels)
        predictions = {
            method: labels_to_indicator(prediction_labels[method]) for method in METHODS
        }
        macro_labels = supported_label_ids(reference)
        expected_macro = tuple(AMP_LABEL_IDS)
        if evaluation == "A2":
            expected_macro = tuple(
                label for label in AMP_LABEL_IDS if label != ORGAN_REMOVAL_LABEL
            )
        if macro_labels != expected_macro:
            raise PairedBootstrapError(
                f"{evaluation} supported-label convention has drifted: {macro_labels}"
            )
        evaluation_manifest = manifest.get("evaluations", {}).get(evaluation, {})
        if (
            int(evaluation_manifest.get("test_n", -1)) != EXPECTED_N[evaluation]
            or tuple(evaluation_manifest.get("macro_label_ids", [])) != macro_labels
            or int(evaluation_manifest.get("macro_label_count", -1)) != len(macro_labels)
        ):
            raise PairedBootstrapError(f"{evaluation} manifest membership/macro convention mismatch")
        output[evaluation] = EvaluationMatrices(
            evaluation=evaluation,
            reference=reference,
            predictions=predictions,
            macro_label_ids=macro_labels,
        )

    _validate_point_metrics(output["A1"], REPO_ROOT / "outputs/metrics/a1/amp_primary_results.csv")
    _validate_point_metrics(output["A2"], REPO_ROOT / "outputs/metrics/a2/amp_primary_results.csv")
    return output


def load_evaluation_b(
    manifest_path: Path = DEFAULT_EVAL_B_MANIFEST,
) -> EvaluationMatrices:
    """Load the frozen A3 human-grounded common substantive set."""

    evaluator = importlib.import_module("src.experiments.18_evaluate_evaluation_b")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise PairedBootstrapError("Evaluation B analysis manifest is not COMPLETE")
    if (
        int(manifest.get("retained_n", -1)) != 61
        or int(manifest.get("substantive_n", -1)) != 55
        or int(manifest.get("abstain_n", -1)) != 6
        or int(manifest.get("common_substantive_n", -1)) != 55
        or list(manifest.get("m4_demo_overlap_case_ids", []))
    ):
        raise PairedBootstrapError("Evaluation B frozen membership/counts have drifted")
    _validate_eval_b_manifest_hashes(manifest)
    evaluator.validate_evaluation_a_integrity(evaluator.DEFAULT_EVAL_A_BASELINE)

    human = evaluator.load_human_reference(evaluator.DEFAULT_HUMAN_REFERENCE)
    source_manifest = _read_json(evaluator.DEFAULT_SOURCE_MANIFEST)
    qc_summary = _read_json(evaluator.DEFAULT_QC_SUMMARY)
    membership_manifest = _read_json(evaluator.DEFAULT_MEMBERSHIP_MANIFEST)
    evaluator.validate_human_reference_provenance(
        source_manifest,
        qc_summary,
        human,
        human_reference_path=evaluator.DEFAULT_HUMAN_REFERENCE,
        source_manifest_path=evaluator.DEFAULT_SOURCE_MANIFEST,
        qc_summary_path=evaluator.DEFAULT_QC_SUMMARY,
        membership_manifest=membership_manifest,
    )
    membership_sha = evaluator.retained_membership_sha256(human)
    if membership_sha != str(manifest.get("retained_membership_sha256", "")):
        raise PairedBootstrapError("Evaluation B retained-membership digest mismatch")
    evaluator.validate_leakage_audit(
        evaluator.DEFAULT_LEAKAGE_AUDIT,
        set(human),
        expected_membership_sha256=membership_sha,
    )
    overlap = evaluator.load_demo_overlap(evaluator.DEFAULT_DEMO_BANK, human)
    if overlap:
        raise PairedBootstrapError(
            "A3 frozen membership unexpectedly contains an M4 demonstration overlap"
        )
    m4_bank_id, m4_bank_membership_sha = evaluator.load_m4_demo_bank_provenance(
        evaluator.DEFAULT_DEMO_BANK
    )
    prediction_paths = dict(evaluator.DEFAULT_PREDICTIONS)
    evaluator.validate_execution_metadata(
        evaluator.DEFAULT_M1_METADATA,
        evaluator.DEFAULT_M2_METADATA,
        evaluator.DEFAULT_M3_DIAGNOSTICS,
        evaluator.DEFAULT_M4_DIAGNOSTICS,
        retained_n=len(human),
        expected_m4_n=len(human),
        demo_overlap_ids=overlap,
        demo_overlap_ranks=set(),
        prediction_paths=prediction_paths,
        expected_membership_sha256=membership_sha,
        expected_m4_demo_bank_id=m4_bank_id,
    )
    predictions: dict[str, dict[str, Any]] = {}
    for method in evaluator.METHODS:
        predictions[method] = evaluator.load_predictions(
            method,
            prediction_paths[method],
            human,
            set(human),
            expected_membership_sha256=membership_sha,
            expected_m4_demo_bank_id=m4_bank_id,
            expected_m4_demo_membership_sha256=m4_bank_membership_sha,
        )
    common_ids, _ = evaluator.build_common_membership(human, predictions, overlap)
    frozen_common = list(manifest.get("common_substantive_case_ids", []))
    if common_ids != frozen_common or len(common_ids) != EXPECTED_N["A3"]:
        raise PairedBootstrapError("A3 common substantive membership/order has drifted")

    reference = evaluator._matrix_from_cases(common_ids, human, kind="human reference")
    method_predictions = {
        method: evaluator._matrix_from_cases(
            common_ids, predictions[method], kind=f"{method} predictions"
        )
        for method in METHODS
    }
    macro_labels = supported_label_ids(reference)
    if macro_labels != tuple(AMP_LABEL_IDS):
        raise PairedBootstrapError(
            f"A3 supported-label convention has drifted: {macro_labels}"
        )
    output = EvaluationMatrices(
        evaluation="A3",
        reference=reference,
        predictions=method_predictions,
        macro_label_ids=macro_labels,
    )
    _validate_point_metrics(
        output,
        REPO_ROOT / "outputs/analysis/evaluation_b/eval_b_main_results.csv",
    )
    return output


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise PairedBootstrapError("Refusing to write an empty paired-bootstrap table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def generate(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    if n_resamples != DEFAULT_BOOTSTRAP_RESAMPLES or seed != DEFAULT_BOOTSTRAP_SEED:
        raise PairedBootstrapError(
            "Paper-final generation requires exactly 1,000 resamples and seed 20260811"
        )
    evaluation_a = load_evaluation_a()
    evaluation_b = load_evaluation_b()
    datasets = (evaluation_a["A1"], evaluation_a["A2"], evaluation_b)
    rows = [
        row
        for data in datasets
        for row in paired_bootstrap_rows(data, n_resamples=n_resamples, seed=seed)
    ]
    expected_rows = len(EXPECTED_N) * len(COMPARISONS) * len(METRIC_DISPLAY)
    if len(rows) != expected_rows:
        raise PairedBootstrapError(
            f"Generated {len(rows)} paired rows; expected {expected_rows}"
        )
    _atomic_write_csv(output_path, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = generate(
            output_path=args.output,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
    except (PairedBootstrapError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "version": VERSION,
                "output": str(args.output),
                "rows": len(rows),
                "evaluations": {evaluation: EXPECTED_N[evaluation] for evaluation in EXPECTED_N},
                "comparisons": [f"{first} - {second}" for first, second in COMPARISONS],
                "bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
                "output_sha256": _sha256_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

