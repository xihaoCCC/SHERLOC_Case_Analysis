# Final Evaluation A report

Status: **AUTHORITATIVE PAPER-ANALYSIS PACKAGE FROM FROZEN PREDICTIONS**  
Generator: `src/experiments/16_finalize_evaluation_a.py` v1.0.0

## 1. Objective

Evaluation A compares four frozen methods for extracting 17 Act/Means/Purpose
(AMP) dimensions from English SHERLOC Fact Summaries. The large-scale targets
are SHERLOC **silver-reference labels**. They are distinct from the later
human-grounded gold annotations.

## 2. A1 IID design

A1 is the frozen IID TEST split with **N=253**. M1 and M2 selection and threshold
choices used only frozen TRAIN/VALIDATION data. M3 and M4 used the already-frozen
zero-shot and six-shot prompts. The final table below copies point estimates and
95% case-bootstrap intervals from the canonical evaluator.

## 3. A2 jurisdiction-OOD design

A2 pools three jurisdiction-disjoint TEST folds: Fold 1 N=288, Fold 2 N=287,
and Fold 3 N=286, for **N=861** unique cases. Each held-out jurisdiction is TEST
in one fold. Demonstration jurisdictions were disjoint from the corresponding
M4 TEST jurisdictions.

## 4. Methods

- **M1:** TF-IDF features with one-vs-rest logistic regression.
- **M2:** `answerdotai/ModernBERT-base` with a 17-logit multilabel head.
- **M3:** `gpt-5.6-luna` zero-shot structured AMP extraction.
- **M4:** the same LLM configuration and extraction instructions as M3 with six
  frozen solved demonstrations appropriate to the split/fold.

## 5. Frozen samples and label space

The primary cohort contains 1,263 cases and the ontology contains 5 Act, 6 Means,
and 6 Purpose labels. A1 Macro-F1 uses all 17 labels. A2 retains all 17 prediction
dimensions, but pooled A2 Macro-F1 averages the 16 labels with positive
silver-reference support because `PURPOSE_REMOVAL_OF_ORGANS` has support zero.

## 6. Primary metrics

Per-label precision, recall, and F1 use the canonical multilabel counts.
Macro-F1 is the unweighted mean over eligible supported labels; Micro-F1 pools
counts across all 17 dimensions. Exact-set accuracy requires equality of the
complete predicted and reference sets. Example Jaccard is the mean case-level
intersection-over-union. Confidence intervals are the frozen 1,000-resample
case bootstrap with seed `20260811`.

## 7. Contained Partial Match Rate

For one AMP family, case-level CPMR is 1 exactly when the predicted set is
nonempty and is a subset of that case's silver-reference set; otherwise it is 0.
Group CPMR is its case mean. CPMR is a secondary diagnostic of
reference-contained prediction behavior, not absolute factual correctness.

## 8. Mean Contained Recall

Contained Recall is the predicted/reference set-size ratio only on CPMR-successful
cases. Mean Contained Recall averages those defined values. It describes
conditional completeness and must be read alongside CPMR, not as a standalone
method ranking.

## 9. Final A1 IID results

| Method | N | Macro-F1 [95% CI] | Micro-F1 [95% CI] | Exact set [95% CI] | Jaccard [95% CI] | Act CPMR/MCR | Means CPMR/MCR | Purpose CPMR/MCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 253 | 0.467 [0.448, 0.483] | 0.715 [0.696, 0.732] | 0.004 [0.000, 0.012] | 0.558 [0.536, 0.578] | 0.150/0.953 | 0.111/0.906 | 0.767/0.956 |
| M2 | 253 | 0.535 [0.502, 0.561] | 0.718 [0.701, 0.735] | 0.012 [0.000, 0.028] | 0.567 [0.546, 0.587] | 0.249/0.913 | 0.138/0.880 | 0.771/0.953 |
| M3 | 253 | 0.593 [0.518, 0.619] | 0.722 [0.700, 0.741] | 0.047 [0.024, 0.075] | 0.579 [0.552, 0.604] | 0.482/0.770 | 0.435/0.764 | 0.759/0.951 |
| M4 | 253 | 0.602 [0.528, 0.628] | 0.721 [0.699, 0.739] | 0.063 [0.036, 0.095] | 0.574 [0.546, 0.600] | 0.506/0.772 | 0.458/0.776 | 0.791/0.961 |

Canonical machine-readable table: [`a1_main_comparison.csv`](../outputs/analysis/evaluation_a/a1_main_comparison.csv).

## 10. Final pooled A2 jurisdiction-OOD results

| Method | N | Macro-F1 [95% CI] | Micro-F1 [95% CI] | Exact set [95% CI] | Jaccard [95% CI] | Act CPMR/MCR | Means CPMR/MCR | Purpose CPMR/MCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 861 | 0.480 [0.465, 0.497] | 0.705 [0.696, 0.714] | 0.001 [0.000, 0.003] | 0.546 [0.536, 0.558] | 0.134/0.961 | 0.146/0.846 | 0.776/0.958 |
| M2 | 861 | 0.527 [0.507, 0.547] | 0.706 [0.697, 0.715] | 0.008 [0.003, 0.014] | 0.546 [0.536, 0.557] | 0.177/0.899 | 0.177/0.786 | 0.847/0.950 |
| M3 | 861 | 0.572 [0.552, 0.590] | 0.729 [0.718, 0.738] | 0.043 [0.029, 0.058] | 0.577 [0.563, 0.591] | 0.510/0.787 | 0.433/0.756 | 0.762/0.965 |
| M4 | 861 | 0.568 [0.548, 0.587] | 0.726 [0.715, 0.737] | 0.050 [0.036, 0.064] | 0.575 [0.560, 0.590] | 0.571/0.766 | 0.459/0.748 | 0.803/0.969 |

Canonical machine-readable table: [`a2_main_comparison.csv`](../outputs/analysis/evaluation_a/a2_main_comparison.csv).

## 11. A1 to A2 descriptive shifts

Each delta is pooled A2 minus A1. No statistical-significance test was designated,
so the table supports descriptive comparison only.

| Method | Δ Macro | Δ Micro | Δ Exact | Δ Jaccard | Δ Act CPMR | Δ Means CPMR | Δ Purpose CPMR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | +0.013 | -0.010 | -0.003 | -0.012 | -0.017 | +0.036 | +0.009 |
| M2 | -0.007 | -0.012 | -0.004 | -0.020 | -0.072 | +0.038 | +0.076 |
| M3 | -0.021 | +0.007 | -0.004 | -0.002 | +0.028 | -0.002 | +0.003 |
| M4 | -0.034 | +0.005 | -0.013 | +0.001 | +0.065 | +0.000 | +0.012 |

## 12. M3 versus M4

Each value is M4 six-shot minus M3 zero-shot. M4 has higher CPMR in all three
families in both evaluations, while the conventional aggregate metrics have
mixed directions and small differences. This is not a uniform superiority claim.

| Evaluation | N | Δ Macro | Δ Micro | Δ Exact | Δ Jaccard | Δ Act CPMR/MCR | Δ Means CPMR/MCR | Δ Purpose CPMR/MCR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 253 | +0.008 | -0.001 | +0.016 | -0.005 | +0.024/+0.002 | +0.024/+0.012 | +0.032/+0.010 |
| A2 | 861 | -0.004 | -0.003 | +0.007 | -0.002 | +0.062/-0.022 | +0.026/-0.008 | +0.041/+0.004 |

Per-label F1 differences, including A2 organ-removal `N/A`, are preserved in
[`m3_vs_m4_per_label_f1.csv`](../outputs/analysis/evaluation_a/m3_vs_m4_per_label_f1.csv).

## 13. Family-level results

These are unweighted means of canonical per-label precision, recall, and F1
within each family. Undefined zero-support labels are excluded, never treated as
numeric zero. Thus A2 Purpose uses five supported labels; the other cells use all
labels in their family.

| Evaluation | Method | Family | Supported labels | Mean P | Mean R | Mean F1 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | M1 | Act | 5 | 0.557 | 0.958 | 0.695 |
| A1 | M1 | Means | 6 | 0.422 | 0.633 | 0.486 |
| A1 | M1 | Purpose | 6 | 0.250 | 0.270 | 0.258 |
| A1 | M2 | Act | 5 | 0.576 | 0.875 | 0.691 |
| A1 | M2 | Means | 6 | 0.405 | 0.655 | 0.495 |
| A1 | M2 | Purpose | 6 | 0.415 | 0.499 | 0.444 |
| A1 | M3 | Act | 5 | 0.664 | 0.666 | 0.658 |
| A1 | M3 | Means | 6 | 0.619 | 0.540 | 0.550 |
| A1 | M3 | Purpose | 6 | 0.558 | 0.618 | 0.583 |
| A1 | M4 | Act | 5 | 0.682 | 0.613 | 0.636 |
| A1 | M4 | Means | 6 | 0.652 | 0.517 | 0.551 |
| A1 | M4 | Purpose | 6 | 0.610 | 0.649 | 0.625 |
| A2 | M1 | Act | 5 | 0.536 | 0.956 | 0.671 |
| A2 | M1 | Means | 6 | 0.409 | 0.577 | 0.451 |
| A2 | M1 | Purpose | 5 | 0.385 | 0.334 | 0.323 |
| A2 | M2 | Act | 5 | 0.549 | 0.884 | 0.668 |
| A2 | M2 | Means | 6 | 0.399 | 0.559 | 0.462 |
| A2 | M2 | Purpose | 5 | 0.494 | 0.458 | 0.466 |
| A2 | M3 | Act | 5 | 0.688 | 0.650 | 0.659 |
| A2 | M3 | Means | 6 | 0.631 | 0.516 | 0.547 |
| A2 | M3 | Purpose | 5 | 0.475 | 0.580 | 0.517 |
| A2 | M4 | Act | 5 | 0.707 | 0.593 | 0.632 |
| A2 | M4 | Means | 6 | 0.642 | 0.498 | 0.539 |
| A2 | M4 | Purpose | 5 | 0.522 | 0.568 | 0.539 |

## 14. Prediction breadth

Breadth is descriptive and is not a primary performance metric. M1/M2 generally
produce broader Act/Means sets, whereas M3/M4 produce narrower sets; this pattern
helps contextualize their different recall and CPMR behavior. Full family-specific
reference means and zero-prediction proportions are in the CSV.

| Evaluation | Method | Mean Act | Mean Means | Mean Purpose | Mean predicted total | Mean silver total | Median [P25, P75] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | M1 | 4.455 | 3.352 | 1.174 | 8.980 | 5.889 | 9.000 [8.000, 10.000] |
| A1 | M2 | 3.941 | 3.407 | 1.292 | 8.640 | 5.889 | 9.000 [8.000, 10.000] |
| A1 | M3 | 2.648 | 1.929 | 1.241 | 5.818 | 5.889 | 6.000 [4.000, 7.000] |
| A1 | M4 | 2.419 | 1.826 | 1.166 | 5.411 | 5.889 | 5.000 [4.000, 7.000] |
| A2 | M1 | 4.598 | 3.377 | 1.204 | 9.180 | 5.947 | 9.000 [9.000, 10.000] |
| A2 | M2 | 4.158 | 3.124 | 1.152 | 8.434 | 5.947 | 9.000 [7.000, 10.000] |
| A2 | M3 | 2.526 | 1.875 | 1.229 | 5.630 | 5.947 | 6.000 [4.000, 7.000] |
| A2 | M4 | 2.264 | 1.810 | 1.141 | 5.214 | 5.947 | 5.000 [4.000, 7.000] |

## 15. Rare-label sensitivity

The diagnostic A1 value excludes `PURPOSE_REMOVAL_OF_ORGANS` (support 2); it does
not replace the official 17-label result. In A2, organ removal has support 0 and
is already excluded from the official 16-supported-label Macro-F1.

| Evaluation | Method | Official Macro-F1 | Diagnostic without organ removal | Difference |
| --- | --- | --- | --- | --- |
| A1 | M1 | 0.467 | 0.496 | +0.029 |
| A1 | M2 | 0.535 | 0.568 | +0.033 |
| A1 | M3 | 0.593 | 0.568 | -0.025 |
| A1 | M4 | 0.602 | 0.577 | -0.025 |
| A2 | M1 | 0.480 | 0.480 | +0.000 |
| A2 | M2 | 0.527 | 0.527 | +0.000 |
| A2 | M3 | 0.572 | 0.572 | +0.000 |
| A2 | M4 | 0.568 | 0.568 | +0.000 |

See [`evaluation_a_rare_label_sensitivity.md`](evaluation_a_rare_label_sensitivity.md)
and the complete per-label support profile in
[`rare_label_sensitivity.csv`](../outputs/analysis/evaluation_a/rare_label_sensitivity.csv).

## 16. Fold and jurisdiction heterogeneity

| Fold | Method | N | Macro-F1 | Micro-F1 | Jaccard | Exact set | Act CPMR | Means CPMR | Purpose CPMR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | M1 | 288 | 0.510 | 0.704 | 0.545 | 0.000 | 0.128 | 0.111 | 0.705 |
| 1 | M2 | 288 | 0.524 | 0.722 | 0.564 | 0.010 | 0.174 | 0.174 | 0.854 |
| 1 | M3 | 288 | 0.598 | 0.760 | 0.626 | 0.073 | 0.500 | 0.507 | 0.795 |
| 1 | M4 | 288 | 0.606 | 0.761 | 0.629 | 0.073 | 0.559 | 0.545 | 0.851 |
| 2 | M1 | 287 | 0.455 | 0.709 | 0.549 | 0.003 | 0.139 | 0.098 | 0.864 |
| 2 | M2 | 287 | 0.520 | 0.696 | 0.534 | 0.000 | 0.164 | 0.115 | 0.857 |
| 2 | M3 | 287 | 0.507 | 0.690 | 0.522 | 0.024 | 0.516 | 0.317 | 0.791 |
| 2 | M4 | 287 | 0.476 | 0.674 | 0.505 | 0.031 | 0.561 | 0.348 | 0.805 |
| 3 | M1 | 286 | 0.453 | 0.701 | 0.543 | 0.000 | 0.133 | 0.231 | 0.759 |
| 3 | M2 | 286 | 0.517 | 0.699 | 0.540 | 0.014 | 0.192 | 0.241 | 0.829 |
| 3 | M3 | 286 | 0.576 | 0.730 | 0.583 | 0.031 | 0.514 | 0.476 | 0.699 |
| 3 | M4 | 286 | 0.576 | 0.736 | 0.590 | 0.045 | 0.594 | 0.483 | 0.752 |

The paper-facing jurisdiction table preserves 18
jurisdictions and all four methods; jurisdiction N ranges from
20 to 160. These results are descriptive.
Do not rank jurisdictions from these cells, and do not overinterpret small-N
differences. See
[`a2_jurisdiction_summary.csv`](../outputs/analysis/evaluation_a/a2_jurisdiction_summary.csv).

## 17. M3/M4 API execution summary

| Evaluation | Method | Successful cases | API attempts | Retries | Recorded tokens | Median latency | P90 latency | Active runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | M3 | 253 | 264 | 11 | 542,416 | 2.747 | 5.161 | N/A |
| A1 | M4 | 253 | 319 | 66 | 957,533 | 2.173 | 4.521 | N/A |
| A2 | M3 | 861 | 922 | 61 | 1,829,456 | 2.264 | 4.414 | 785s |
| A2 | M4 | 861 | 988 | 127 | 3,183,854 | 1.939 | 3.663 | 856s |

A2 token totals reflect recorded successful-response usage and are lower bounds
when failed-request usage was unavailable. Technical retry policies, including
the one-case M4 A2 rank-1340 rate-limit exception, changed no model, prompt,
schema, demonstration, target text, or scoring rule.

## 18. Methodological cautions and limitations

- SHERLOC labels are silver-reference labels; later adjudicated annotations will
  be human-grounded gold.
- CPMR captures reference-contained behavior. It does not establish factual or
  legal correctness when the silver reference is incomplete or mismatched to the
  narrative.
- A1 and A2 compare different case sets; reported deltas have no designated
  significance test.
- A2 per-jurisdiction cells can be small and are descriptive only.
- The two A1 organ-removal cases make its per-label F1 and Macro-F1 contribution
  unusually sensitive; the canonical metric remains unchanged.
- Purpose behaves differently from Act and Means, including much higher CPMR for
  every method, so a single aggregate characterization can obscure family-level
  behavior.
- The four methods do not have one uniform ordering across Macro-F1, Micro-F1,
  exact set, Jaccard, CPMR, and conditional contained recall.

## Core figures

1. [`A1 versus A2 core performance`](../outputs/figures/evaluation_a/figure_1_a1_vs_a2_core_performance.svg)
2. [`CPMR by AMP family`](../outputs/figures/evaluation_a/figure_2_cpmr_by_amp_family.svg)
3. [`CPMR versus Mean Contained Recall`](../outputs/figures/evaluation_a/figure_3_cpmr_vs_contained_recall.svg)
4. [`Per-label F1`](../outputs/figures/evaluation_a/figure_4_per_label_f1.svg)
