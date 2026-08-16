# Evaluation A rare-label sensitivity

Status: **DESCRIPTIVE SENSITIVITY ANALYSIS; NOT A REPLACEMENT FOR THE CANONICAL METRIC**  
Generator: `src/experiments/16_finalize_evaluation_a.py` v1.0.0

## Why this diagnostic is reported

Macro-F1 assigns equal weight to every eligible label. Consequently, an ultra-rare
label can materially change the arithmetic mean even though it affects very few
cases. This is a property of Macro-F1, not evidence that the rare label should be
removed from the frozen task.

`PURPOSE_REMOVAL_OF_ORGANS` has silver-reference support **2** in A1. The official
A1 Macro-F1 therefore uses all 17 frozen AMP labels. The diagnostic below excludes
that single label and averages the other 16 canonical per-label F1 values. It does
not change the official result.

| Method | Official 17-label Macro-F1 | Diagnostic 16-label Macro-F1 | Diagnostic − official |
| --- | --- | --- | --- |
| M1 | 0.466676 | 0.495843 | +0.029167 |
| M2 | 0.534616 | 0.568029 | +0.033413 |
| M3 | 0.593421 | 0.568010 | -0.025411 |
| M4 | 0.601718 | 0.576825 | -0.024893 |

The A1 organ-removal per-label F1 is 0 for M1/M2 and 1 for M3/M4 on two supported
cases. This extreme small-support result explains why excluding the label moves
the methods' macro averages in different directions; it should not be generalized.

## A2 zero-support rule

In pooled A2, `PURPOSE_REMOVAL_OF_ORGANS` has silver-reference support **0**. It
remains one of the 17 prediction dimensions, so an organ-removal false positive
would still affect all-label micro and set metrics. Its per-label precision,
recall, and F1 are `N/A`, and it is already excluded from the official A2
Macro-F1, which averages the 16 positive-support labels.

| Method | Official supported-label Macro-F1 | Supported labels |
| --- | --- | --- |
| M1 | 0.479845 | 16 |
| M2 | 0.527488 | 16 |
| M3 | 0.572257 | 16 |
| M4 | 0.568094 | 16 |

The complete two-evaluation support profile for every AMP label is preserved in
[`rare_label_sensitivity.csv`](../outputs/analysis/evaluation_a/rare_label_sensitivity.csv).

## Interpretation boundary

This analysis is descriptive. It was computed after the predictions were frozen,
does not alter any model or threshold, and must not replace the canonical
Macro-F1. All reference labels discussed here are SHERLOC silver-reference labels,
not human-grounded gold.
