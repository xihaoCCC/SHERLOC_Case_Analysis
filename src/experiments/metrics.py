#!/usr/bin/env python3
"""Canonical metrics for the 17-label SHERLOC Legacy AMP benchmark.

The large-scale SHERLOC targets are *silver-reference labels*.  This module
uses that terminology throughout its public API so downstream analysis does
not accidentally describe the Legacy Keywords as human-adjudicated reference
annotations.

The same functions are intended for M1--M4 and both Evaluation A1 and A2.
Geographic Form and all auxiliary targets are deliberately out of scope.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


AMP_LABEL_IDS: tuple[str, ...] = (
    "ACT_RECRUITMENT",
    "ACT_TRANSPORTATION",
    "ACT_TRANSFER",
    "ACT_HARBOURING",
    "ACT_RECEIPT",
    "MEANS_THREAT_FORCE_OR_COERCION",
    "MEANS_ABDUCTION",
    "MEANS_FRAUD",
    "MEANS_DECEPTION",
    "MEANS_ABUSE_OF_POWER_OR_VULNERABILITY",
    "MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL",
    "PURPOSE_SEXUAL_EXPLOITATION",
    "PURPOSE_FORCED_LABOUR_OR_SERVICES",
    "PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES",
    "PURPOSE_SERVITUDE",
    "PURPOSE_REMOVAL_OF_ORGANS",
    "PURPOSE_OTHER",
)

AMP_FAMILY_BY_LABEL: dict[str, str] = {
    **{label: "ACT" for label in AMP_LABEL_IDS[:5]},
    **{label: "MEANS" for label in AMP_LABEL_IDS[5:11]},
    **{label: "PURPOSE" for label in AMP_LABEL_IDS[11:]},
}

ORGAN_REMOVAL_LABEL = "PURPOSE_REMOVAL_OF_ORGANS"
SILVER_REFERENCE_TERM = "SHERLOC Legacy Keywords silver reference"
AMP_FAMILIES: tuple[str, ...] = ("ACT", "MEANS", "PURPOSE")


class MetricInputError(ValueError):
    """Raised when a prediction artifact cannot be evaluated safely."""


@dataclass(frozen=True)
class BinaryMatrices:
    """Validated, aligned binary matrices used by the metric functions."""

    silver_reference: np.ndarray
    prediction: np.ndarray
    label_ids: tuple[str, ...]


def _binary_matrix(value: Any, *, name: str, n_labels: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise MetricInputError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] == 0:
        raise MetricInputError(f"{name} must contain at least one test case")
    if array.shape[1] != n_labels:
        raise MetricInputError(
            f"{name} has {array.shape[1]} columns; expected {n_labels}"
        )
    if array.dtype == np.bool_:
        return array.astype(np.uint8, copy=False)
    try:
        numeric = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise MetricInputError(f"{name} is not numeric/binary") from exc
    if not np.isfinite(numeric).all():
        raise MetricInputError(f"{name} contains non-finite values")
    if not np.isin(numeric, (0.0, 1.0)).all():
        raise MetricInputError(f"{name} must contain only 0/1 values")
    return numeric.astype(np.uint8, copy=False)


def validate_binary_matrices(
    silver_reference: Any,
    prediction: Any,
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
) -> BinaryMatrices:
    """Validate and align two case-by-label binary matrices."""

    labels = tuple(label_ids)
    if len(labels) != len(set(labels)):
        raise MetricInputError("label_ids contains duplicates")
    if labels != AMP_LABEL_IDS:
        unknown = set(labels) - set(AMP_LABEL_IDS)
        if unknown:
            raise MetricInputError(f"Unknown AMP label IDs: {sorted(unknown)}")
    reference = _binary_matrix(
        silver_reference, name="silver_reference", n_labels=len(labels)
    )
    predicted = _binary_matrix(prediction, name="prediction", n_labels=len(labels))
    if reference.shape != predicted.shape:
        raise MetricInputError(
            "silver_reference and prediction matrices must have identical shapes"
        )
    return BinaryMatrices(reference, predicted, labels)


def labels_to_indicator(
    rows: Sequence[Iterable[str] | Mapping[str, Any] | Sequence[int]],
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
) -> np.ndarray:
    """Convert label-ID collections, binary maps, or 0/1 vectors to a matrix.

    A sequence of exactly 17 numeric/boolean values is interpreted as a vector.
    Other sequences are interpreted as collections of ontology IDs.  Unknown
    ontology IDs are rejected rather than silently discarded.
    """

    labels = tuple(label_ids)
    label_index = {label: index for index, label in enumerate(labels)}
    result = np.zeros((len(rows), len(labels)), dtype=np.uint8)
    for row_index, raw in enumerate(rows):
        if isinstance(raw, Mapping):
            unknown = set(raw) - set(labels)
            if unknown:
                raise MetricInputError(
                    f"Row {row_index} contains unknown AMP labels: {sorted(unknown)}"
                )
            for label, value in raw.items():
                if value not in (0, 1, False, True):
                    raise MetricInputError(
                        f"Row {row_index} map value for {label} is not binary"
                    )
                result[row_index, label_index[label]] = int(value)
            continue

        values = list(raw)
        is_vector = len(values) == len(labels) and all(
            value in (0, 1, False, True) for value in values
        )
        if is_vector:
            result[row_index] = np.asarray(values, dtype=np.uint8)
            continue

        unknown = set(values) - set(labels)
        if unknown:
            raise MetricInputError(
                f"Row {row_index} contains unknown AMP labels: {sorted(unknown)}"
            )
        for label in values:
            result[row_index, label_index[label]] = 1
    return result


def indicator_to_labels(
    matrix: Any, *, label_ids: Sequence[str] = AMP_LABEL_IDS
) -> list[list[str]]:
    """Convert an aligned binary indicator matrix to ordered label-ID lists."""

    labels = tuple(label_ids)
    validated = _binary_matrix(matrix, name="matrix", n_labels=len(labels))
    return [
        [label for label, value in zip(labels, row, strict=True) if value]
        for row in validated
    ]


def supported_label_ids(
    silver_reference: Any, *, label_ids: Sequence[str] = AMP_LABEL_IDS
) -> tuple[str, ...]:
    """Return labels with at least one positive silver-reference test case."""

    labels = tuple(label_ids)
    reference = _binary_matrix(
        silver_reference, name="silver_reference", n_labels=len(labels)
    )
    support = reference.sum(axis=0)
    return tuple(label for label, count in zip(labels, support, strict=True) if count)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1_zero_division_zero(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float((2 * tp) / denominator) if denominator else 0.0


def contained_partial_match(
    silver_reference_labels: Iterable[str], predicted_labels: Iterable[str]
) -> int:
    """Return 1 for a nonempty prediction contained in the reference set.

    Labels are compared as sets, so input order and duplicate presentation do
    not affect the result. An empty prediction always returns 0. Consequently,
    an empty silver-reference set also returns 0 safely.
    """

    reference_set = frozenset(silver_reference_labels)
    predicted_set = frozenset(predicted_labels)
    return int(bool(predicted_set) and predicted_set.issubset(reference_set))


def contained_recall(
    silver_reference_labels: Iterable[str], predicted_labels: Iterable[str]
) -> float | None:
    """Return reference coverage only for CPMR successes; otherwise N/A."""

    reference_set = frozenset(silver_reference_labels)
    predicted_set = frozenset(predicted_labels)
    if not predicted_set or not predicted_set.issubset(reference_set):
        return None
    return float(len(predicted_set) / len(reference_set))


def compute_amp_cpmr(
    silver_reference: Any,
    prediction: Any,
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
) -> dict[str, Any]:
    """Calculate independent Act, Means, and Purpose CPMR diagnostics.

    Existing canonical mappings must be applied before this matrix-level
    function, as they are for the other shared AMP metrics. Per-case contained
    recall is ``None`` unless the corresponding family CPMR is 1; a family
    with no CPMR successes also has ``mean_contained_recall=None``.
    """

    matrices = validate_binary_matrices(
        silver_reference, prediction, label_ids=label_ids
    )
    case_n = matrices.silver_reference.shape[0]
    per_case: list[dict[str, int | float | None]] = [{} for _ in range(case_n)]
    by_family: dict[str, dict[str, int | float | None]] = {}

    for family in AMP_FAMILIES:
        indices = [
            index
            for index, label in enumerate(matrices.label_ids)
            if AMP_FAMILY_BY_LABEL[label] == family
        ]
        family_reference = matrices.silver_reference[:, indices]
        family_prediction = matrices.prediction[:, indices]
        success = np.logical_and(
            family_prediction.sum(axis=1) > 0,
            np.all(family_prediction <= family_reference, axis=1),
        )
        reference_count = family_reference.sum(axis=1)
        predicted_count = family_prediction.sum(axis=1)
        recalls = np.divide(
            predicted_count,
            reference_count,
            out=np.zeros(case_n, dtype=np.float64),
            where=success,
        )

        family_key = family.lower()
        for case_index, is_success in enumerate(success):
            per_case[case_index][f"{family_key}_cpmr"] = int(is_success)
            per_case[case_index][f"{family_key}_contained_recall"] = (
                float(recalls[case_index]) if is_success else None
            )

        success_count = int(success.sum())
        by_family[family] = {
            "cpmr": float(success.mean()),
            "mean_contained_recall": (
                float(recalls[success].mean()) if success_count else None
            ),
            "success_count": success_count,
        }

    return {"test_n": int(case_n), "by_family": by_family, "per_case": per_case}


def compute_amp_metrics(
    silver_reference: Any,
    prediction: Any,
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
    macro_label_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Calculate canonical aggregate and per-label AMP metrics.

    ``macro_label_ids`` fixes the dimensions included in macro-F1.  If omitted,
    it is derived once from labels with positive support in the supplied test
    universe.  The A2 evaluator passes the 16 labels supported in the complete
    pooled OOD test universe to every fold and bootstrap resample.

    A zero-support label is reported with ``status=NO_REFERENCE_SUPPORT`` and
    ``recall``/``f1`` set to ``None`` (machine-readable N/A).  Its false-positive
    dimensions still affect micro-F1, exact-set accuracy, and Jaccard.  If a
    fixed macro label happens to receive zero support within a bootstrap
    resample, its resampled F1 contribution follows the conventional
    zero-division value of 0; this keeps the bootstrap estimand fixed.
    """

    matrices = validate_binary_matrices(
        silver_reference, prediction, label_ids=label_ids
    )
    reference = matrices.silver_reference
    predicted = matrices.prediction
    labels = matrices.label_ids

    tp = np.logical_and(reference == 1, predicted == 1).sum(axis=0)
    fp = np.logical_and(reference == 0, predicted == 1).sum(axis=0)
    fn = np.logical_and(reference == 1, predicted == 0).sum(axis=0)
    support = reference.sum(axis=0)
    predicted_positive = predicted.sum(axis=0)

    if macro_label_ids is None:
        macro_labels = tuple(
            label for label, count in zip(labels, support, strict=True) if count
        )
    else:
        macro_labels = tuple(macro_label_ids)
        unknown = set(macro_labels) - set(labels)
        if unknown:
            raise MetricInputError(
                f"macro_label_ids contains unknown labels: {sorted(unknown)}"
            )
        if len(macro_labels) != len(set(macro_labels)):
            raise MetricInputError("macro_label_ids contains duplicates")
    if not macro_labels:
        raise MetricInputError("Macro-F1 requires at least one included label")

    per_label: list[dict[str, Any]] = []
    f1_by_label: dict[str, float] = {}
    for index, label in enumerate(labels):
        label_tp = int(tp[index])
        label_fp = int(fp[index])
        label_fn = int(fn[index])
        label_support = int(support[index])
        precision = (
            _safe_ratio(label_tp, label_tp + label_fp) if label_support else None
        )
        if label_support and precision is None:
            # A supported label with no positive predictions has conventional
            # zero precision, not an unavailable metric.
            precision = 0.0
        recall = _safe_ratio(label_tp, label_support)
        display_f1 = (
            _f1_zero_division_zero(label_tp, label_fp, label_fn)
            if label_support
            else None
        )
        f1_by_label[label] = _f1_zero_division_zero(
            label_tp, label_fp, label_fn
        )
        per_label.append(
            {
                "label_id": label,
                "family": AMP_FAMILY_BY_LABEL[label],
                "support": label_support,
                "predicted_positive": int(predicted_positive[index]),
                "true_positive": label_tp,
                "false_positive": label_fp,
                "false_negative": label_fn,
                "precision": precision,
                "recall": recall,
                "f1": display_f1,
                "status": "SUPPORTED" if label_support else "NO_REFERENCE_SUPPORT",
                "included_in_macro_f1": label in macro_labels,
            }
        )

    macro_f1 = float(np.mean([f1_by_label[label] for label in macro_labels]))
    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    total_fn = int(fn.sum())
    micro_f1 = _f1_zero_division_zero(total_tp, total_fp, total_fn)

    exact_by_case = np.all(reference == predicted, axis=1)
    intersection = np.logical_and(reference == 1, predicted == 1).sum(axis=1)
    union = np.logical_or(reference == 1, predicted == 1).sum(axis=1)
    # Empty reference + empty prediction is a perfect set match.
    jaccard_by_case = np.divide(
        intersection,
        union,
        out=np.ones(reference.shape[0], dtype=np.float64),
        where=union != 0,
    )

    return {
        "silver_reference_terminology": SILVER_REFERENCE_TERM,
        "test_n": int(reference.shape[0]),
        "label_count": len(labels),
        "macro_label_count": len(macro_labels),
        "macro_label_ids": list(macro_labels),
        "zero_reference_support_label_ids": [
            label for label, count in zip(labels, support, strict=True) if not count
        ],
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "exact_set_accuracy": float(exact_by_case.mean()),
        "example_jaccard": float(jaccard_by_case.mean()),
        "micro_true_positive": total_tp,
        "micro_false_positive": total_fp,
        "micro_false_negative": total_fn,
        "per_label": per_label,
    }


def compute_case_errors(
    silver_reference: Any,
    prediction: Any,
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
) -> list[dict[str, Any]]:
    """Return ordered silver/predicted labels and errors for every test case."""

    matrices = validate_binary_matrices(
        silver_reference, prediction, label_ids=label_ids
    )
    rows: list[dict[str, Any]] = []
    for reference_row, predicted_row in zip(
        matrices.silver_reference, matrices.prediction, strict=True
    ):
        reference_labels = [
            label
            for label, value in zip(matrices.label_ids, reference_row, strict=True)
            if value
        ]
        predicted_labels = [
            label
            for label, value in zip(matrices.label_ids, predicted_row, strict=True)
            if value
        ]
        reference_set = set(reference_labels)
        predicted_set = set(predicted_labels)
        union = reference_set | predicted_set
        rows.append(
            {
                "silver_reference_labels": reference_labels,
                "predicted_labels": predicted_labels,
                "false_positive_labels": [
                    label for label in matrices.label_ids if label in predicted_set - reference_set
                ],
                "false_negative_labels": [
                    label for label in matrices.label_ids if label in reference_set - predicted_set
                ],
                "exact_set_correct": int(reference_set == predicted_set),
                "example_jaccard": (
                    len(reference_set & predicted_set) / len(union) if union else 1.0
                ),
            }
        )
    return rows


__all__ = [
    "AMP_FAMILIES",
    "AMP_FAMILY_BY_LABEL",
    "AMP_LABEL_IDS",
    "BinaryMatrices",
    "MetricInputError",
    "ORGAN_REMOVAL_LABEL",
    "SILVER_REFERENCE_TERM",
    "compute_amp_cpmr",
    "compute_amp_metrics",
    "compute_case_errors",
    "contained_partial_match",
    "contained_recall",
    "indicator_to_labels",
    "labels_to_indicator",
    "supported_label_ids",
    "validate_binary_matrices",
]
