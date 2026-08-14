#!/usr/bin/env python3
"""Deterministic case-resampling confidence intervals for AMP metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:  # Support both package imports and direct script execution.
    from .metrics import (
        AMP_LABEL_IDS,
        compute_amp_metrics,
        supported_label_ids,
        validate_binary_matrices,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI execution.
    from metrics import (  # type: ignore
        AMP_LABEL_IDS,
        compute_amp_metrics,
        supported_label_ids,
        validate_binary_matrices,
    )


DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_BOOTSTRAP_SEED = 20260811
DEFAULT_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "macro_f1",
    "micro_f1",
    "exact_set_accuracy",
    "example_jaccard",
)


class BootstrapInputError(ValueError):
    """Raised when a confidence interval request is not well defined."""


def percentile_bootstrap_confidence_intervals(
    silver_reference: Any,
    prediction: Any,
    *,
    label_ids: Sequence[str] = AMP_LABEL_IDS,
    macro_label_ids: Sequence[str] | None = None,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, dict[str, Any]]:
    """Return deterministic percentile CIs after resampling test cases.

    The unit of resampling is the complete test case (all 17 dimensions move
    together).  The macro-label set is established from the complete supplied
    test universe unless explicitly provided, then held fixed for all
    resamples.  This is essential for A2's preregistered 16-label macro-F1.
    """

    if not isinstance(n_resamples, int) or n_resamples <= 0:
        raise BootstrapInputError("n_resamples must be a positive integer")
    if not isinstance(seed, int):
        raise BootstrapInputError("seed must be an integer")
    if not 0.0 < confidence_level < 1.0:
        raise BootstrapInputError("confidence_level must lie strictly between 0 and 1")

    matrices = validate_binary_matrices(
        silver_reference, prediction, label_ids=label_ids
    )
    if macro_label_ids is None:
        fixed_macro_labels = supported_label_ids(
            matrices.silver_reference, label_ids=matrices.label_ids
        )
    else:
        fixed_macro_labels = tuple(macro_label_ids)

    estimate = compute_amp_metrics(
        matrices.silver_reference,
        matrices.prediction,
        label_ids=matrices.label_ids,
        macro_label_ids=fixed_macro_labels,
    )
    samples = {
        metric: np.empty(n_resamples, dtype=np.float64)
        for metric in BOOTSTRAP_METRICS
    }
    rng = np.random.default_rng(seed)
    case_n = matrices.silver_reference.shape[0]
    for sample_index in range(n_resamples):
        indices = rng.integers(0, case_n, size=case_n)
        result = compute_amp_metrics(
            matrices.silver_reference[indices],
            matrices.prediction[indices],
            label_ids=matrices.label_ids,
            macro_label_ids=fixed_macro_labels,
        )
        for metric in BOOTSTRAP_METRICS:
            samples[metric][sample_index] = result[metric]

    alpha = (1.0 - confidence_level) / 2.0
    lower_percentile = 100.0 * alpha
    upper_percentile = 100.0 * (1.0 - alpha)
    output: dict[str, dict[str, Any]] = {}
    for metric in BOOTSTRAP_METRICS:
        lower, upper = np.percentile(
            samples[metric], [lower_percentile, upper_percentile], method="linear"
        )
        output[metric] = {
            "estimate": float(estimate[metric]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "confidence_level": confidence_level,
            "n_resamples": n_resamples,
            "seed": seed,
            "method": "case_resampling_percentile_linear",
            "macro_label_count": len(fixed_macro_labels),
            "macro_label_ids": list(fixed_macro_labels),
        }
    return output


__all__ = [
    "BOOTSTRAP_METRICS",
    "BootstrapInputError",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CONFIDENCE_LEVEL",
    "percentile_bootstrap_confidence_intervals",
]
