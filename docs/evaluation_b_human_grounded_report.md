# Evaluation B: single-reviewer human-grounded narrative validation

## 1. Objective and design

Evaluation B assesses AMP extraction against a **single-reviewer human-grounded narrative reference** restricted to information recoverable from each English Fact Summary. Primary Evaluation A remains unchanged and uses SHERLOC Legacy Keywords as silver-reference labels.

Only one human reviewer was available for the completed evaluation, so reviewer-to-reviewer agreement could not be estimated. No second reviewer was fabricated and no reviewer adjudication was performed.

## 2. Human annotation protocol and retained cases

The immutable source contained **100** rows; **74** were reviewed, **26** remained unreviewed, and **13** reviewed cases were marked Skip according to the QC summary. Skip cases were excluded without replacement. The retained reference contains **61** cases: **55** substantive and **6** narrative-insufficiency cases.

Skip indicates that the reviewer judged a case unsuitable for this evaluation. Abstain indicates that the narrative was insufficient for reliable extraction; those cases are retained for a separate diagnostic and are not treated as ordinary all-negative AMP references.

## 3. Human-reference construction and AMP support

Human labels were deterministically syntax-normalized and mapped to the frozen AMP ontology without semantic reinterpretation. Organized Criminal Group was separated from Geographic Form. The final primary comparison uses one common substantive membership of **55** cases.

| family | label_id | human_support |
|---|---|---|
| ACT | ACT_RECRUITMENT | 43 |
| ACT | ACT_TRANSPORTATION | 32 |
| ACT | ACT_TRANSFER | 11 |
| ACT | ACT_HARBOURING | 39 |
| ACT | ACT_RECEIPT | 33 |
| MEANS | MEANS_THREAT_FORCE_OR_COERCION | 38 |
| MEANS | MEANS_ABDUCTION | 5 |
| MEANS | MEANS_FRAUD | 6 |
| MEANS | MEANS_DECEPTION | 28 |
| MEANS | MEANS_ABUSE_OF_POWER_OR_VULNERABILITY | 42 |
| MEANS | MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL | 6 |
| PURPOSE | PURPOSE_SEXUAL_EXPLOITATION | 39 |
| PURPOSE | PURPOSE_FORCED_LABOUR_OR_SERVICES | 13 |
| PURPOSE | PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES | 3 |
| PURPOSE | PURPOSE_SERVITUDE | 3 |
| PURPOSE | PURPOSE_REMOVAL_OF_ORGANS | 2 |
| PURPOSE | PURPOSE_OTHER | 4 |

## 4. SHERLOC silver reference versus human narrative reference

| family | n | substantive_n | comparable_n | silver_reference_unavailable_n | exact_set_concordance | mean_jaccard | shared_label_count | silver_only_label_count | human_only_label_count | silver_only_rate_of_silver_labels | human_only_rate_of_human_labels | shared_union_label_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ACT | 54 | 55 | 54 | 1 | 0.389 | 0.715 | 120 | 19 | 38 | 0.137 | 0.241 | 0.678 |
| MEANS | 54 | 55 | 54 | 1 | 0.685 | 0.840 | 108 | 7 | 17 | 0.061 | 0.136 | 0.818 |
| PURPOSE | 55 | 55 | 55 | 0 | 0.818 | 0.909 | 61 | 8 | 3 | 0.116 | 0.047 | 0.847 |

Family-specific concordance denominators exclude structurally absent Legacy Keyword families. Unavailable Act or Means metadata is retained in the case audit with `SILVER_REFERENCE_UNAVAILABLE`; it is not scored as an empty label set.

The largest label-level mismatch counts were:

| family | label_id | silver_only | human_only | shared |
|---|---|---|---|---|
| ACT | ACT_RECEIPT | 0 | 17 | 16 |
| ACT | ACT_TRANSFER | 10 | 2 | 9 |
| ACT | ACT_RECRUITMENT | 5 | 6 | 37 |
| ACT | ACT_TRANSPORTATION | 4 | 5 | 27 |
| ACT | ACT_HARBOURING | 0 | 8 | 31 |
| MEANS | MEANS_DECEPTION | 1 | 6 | 22 |
| MEANS | MEANS_THREAT_FORCE_OR_COERCION | 1 | 6 | 32 |
| MEANS | MEANS_ABUSE_OF_POWER_OR_VULNERABILITY | 1 | 5 | 37 |
| PURPOSE | PURPOSE_FORCED_LABOUR_OR_SERVICES | 4 | 0 | 13 |
| MEANS | MEANS_FRAUD | 3 | 0 | 6 |

Silver-only labels are not automatically errors. SHERLOC structured metadata may contain information broader than what is directly recoverable from the Fact Summary, while human annotation is intentionally narrative-restricted.

## 5. Leakage-free model evaluation design

All retained human cases were required to be excluded from dedicated M1/M2 training, validation, threshold tuning, and supervised selection. M1/M2 used transferred fixed Evaluation A settings; human labels were not used for model tuning. M3/M4 use their frozen AMP-only prompts and technical policy. Any overlap with the active M4 demonstration bank is removed from the common comparison without selecting replacement demonstrations.

The dedicated M1 training N was **1209** and the dedicated M2 training N was **1209**. M3 recorded **30** new successful requests and **31** identical-request reuses; M4 recorded **53** and **8**, respectively.

M1 fixed settings: `{"C":1.0,"class_weight":null,"global_threshold":0.25,"min_df":2,"tfidf_ngram_range":[1,2]}`. M2 fixed settings: `{"epochs":6,"global_threshold":0.2,"learning_rate":3e-05,"weight_decay":0.01}`, with technical execution `{"adamw_foreach":false,"effective_train_batch_size":16,"gradient_accumulation_steps":16,"gradient_checkpointing":true,"gradient_scaler_enabled":false,"max_length":2048,"mixed_precision_dtype":"bfloat16","pad_to_multiple_of":64,"physical_train_batch_size":1}`. M3/M4 used model `gpt-5.6-luna`, their frozen prompt hashes, the frozen schema hash `d106c4ab1aa5bfcf34a6accd4f8c77df0bd21436cb0761d7828b21d9d87f46da`, and `store=false`.

## 6. Main human-grounded M1-M4 results

| method | n | macro_f1 | micro_f1 | exact_set | jaccard | act_cpmr | means_cpmr | purpose_cpmr |
|---|---|---|---|---|---|---|---|---|
| M1 | 55 | 0.456 | 0.731 | 0.000 | 0.578 | 0.073 | 0.236 | 0.673 |
| M2 | 55 | 0.644 | 0.747 | 0.000 | 0.596 | 0.200 | 0.273 | 0.727 |
| M3 | 55 | 0.649 | 0.774 | 0.055 | 0.621 | 0.545 | 0.527 | 0.782 |
| M4 | 55 | 0.612 | 0.764 | 0.018 | 0.603 | 0.582 | 0.564 | 0.836 |

Confidence intervals use 1000 deterministic case-level percentile bootstrap resamples with seed 20260811. The small human-reference N warrants cautious interpretation; no statistical-significance claim is made.

## 7. CPMR, contained recall, and empty-reference behavior

Standard CPMR retains the previously frozen definition: a nonempty prediction contained in the reference set. CPMR_nonempty_reference is reported separately on cases whose human family reference is nonempty. Empty-reference correct-empty rate measures whether a method leaves a family empty when the human narrative reference is empty. CPMR measures reference-contained behavior, not absolute factual correctness.

| method | family | macro_precision_family | macro_recall_family | macro_f1_family | cpmr | mean_contained_recall | cpmr_nonempty_reference | nonempty_reference_n | empty_reference_n | empty_reference_correct_empty_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | ACT | 0.597 | 0.964 | 0.709 | 0.073 | 1.000 | 0.074 | 54 | 1 | 0.000 |
| M1 | MEANS | 0.347 | 0.556 | 0.422 | 0.236 | 0.900 | 0.241 | 54 | 1 | 0.000 |
| M1 | PURPOSE | 0.259 | 0.308 | 0.279 | 0.673 | 0.973 | 0.673 | 55 | 0 | N/A |
| M2 | ACT | 0.642 | 0.839 | 0.702 | 0.200 | 0.885 | 0.204 | 54 | 1 | 0.000 |
| M2 | MEANS | 0.476 | 0.605 | 0.521 | 0.273 | 0.828 | 0.278 | 54 | 1 | 0.000 |
| M2 | PURPOSE | 0.721 | 0.776 | 0.718 | 0.727 | 0.950 | 0.727 | 55 | 0 | N/A |
| M3 | ACT | 0.739 | 0.647 | 0.668 | 0.545 | 0.710 | 0.556 | 54 | 1 | 1.000 |
| M3 | MEANS | 0.617 | 0.598 | 0.588 | 0.527 | 0.820 | 0.537 | 54 | 1 | 1.000 |
| M3 | PURPOSE | 0.668 | 0.724 | 0.693 | 0.782 | 0.988 | 0.782 | 55 | 0 | N/A |
| M4 | ACT | 0.761 | 0.582 | 0.635 | 0.582 | 0.666 | 0.593 | 54 | 1 | 1.000 |
| M4 | MEANS | 0.614 | 0.530 | 0.555 | 0.564 | 0.775 | 0.574 | 54 | 1 | 1.000 |
| M4 | PURPOSE | 0.647 | 0.656 | 0.649 | 0.836 | 0.978 | 0.836 | 55 | 0 | N/A |

## 8. Prediction breadth

| method | mean_predicted_act_labels | mean_predicted_means_labels | mean_predicted_purpose_labels | mean_total_predicted_labels | mean_total_human_labels | silver_act_reference_available_n | silver_means_reference_available_n | silver_purpose_reference_available_n | complete_silver_amp_reference_n | mean_total_silver_labels |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 4.655 | 3.327 | 1.236 | 9.218 | 6.309 | 54 | 54 | 55 | 54 | 5.944 |
| M2 | 3.764 | 3.109 | 1.382 | 8.255 | 6.309 | 54 | 54 | 55 | 54 | 5.944 |
| M3 | 2.527 | 2.055 | 1.236 | 5.818 | 6.309 | 54 | 54 | 55 | 54 | 5.944 |
| M4 | 2.218 | 1.891 | 1.145 | 5.255 | 6.309 | 54 | 54 | 55 | 54 | 5.944 |

## 9. Narrative-insufficiency cases

| method | abstain_n | all_amp_empty_rate | mean_total_predicted_label_count | total_unsupported_predicted_label_count_under_abstention_interpretation |
|---|---|---|---|---|
| M1 | 6 | 0.000 | 8.500 | 51 |
| M2 | 6 | 0.000 | 7.833 | 47 |
| M3 | 6 | 1.000 | 0.000 | 0 |
| M4 | 6 | 1.000 | 0.000 | 0 |

The narrative_insufficiency_safe_rate is an operational descriptive diagnostic, not standard accuracy. M1/M2 were not trained with an explicit abstention objective.

## 10. Silver-scored versus human-scored behavior

`model_silver_vs_human_metric_comparison.csv` reports human-grounded minus silver-reference deltas on the identical **54** common cases with complete silver AMP reference. The primary human-grounded model comparison remains **55** cases; incomplete silver fields are never interpreted as affirmative empty targets. The core numeric deltas are:

| method | metric_scope | metric | dual_reference_n | excluded_incomplete_silver_reference_n | silver_reference_value | human_grounded_value | delta_human_minus_silver | silver_reference_dense_rank | human_grounded_dense_rank | rank_changed |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | OVERALL | macro_f1 | 54 | 1 | 0.438 | 0.460 | 0.022 | 4 | 4 | 0 |
| M1 | OVERALL | micro_f1 | 54 | 1 | 0.694 | 0.738 | 0.043 | 4 | 4 | 0 |
| M1 | OVERALL | exact_set | 54 | 1 | 0.000 | 0.000 | 0.000 | 2 | 3 | 1 |
| M1 | OVERALL | jaccard | 54 | 1 | 0.533 | 0.587 | 0.054 | 4 | 4 | 0 |
| M1 | ACT | cpmr | 54 | 1 | 0.093 | 0.074 | -0.019 | 4 | 4 | 0 |
| M1 | MEANS | cpmr | 54 | 1 | 0.167 | 0.241 | 0.074 | 3 | 4 | 1 |
| M1 | PURPOSE | cpmr | 54 | 1 | 0.685 | 0.685 | 0.000 | 4 | 4 | 0 |
| M2 | OVERALL | macro_f1 | 54 | 1 | 0.592 | 0.645 | 0.053 | 1 | 2 | 1 |
| M2 | OVERALL | micro_f1 | 54 | 1 | 0.713 | 0.749 | 0.035 | 2 | 3 | 1 |
| M2 | OVERALL | exact_set | 54 | 1 | 0.019 | 0.000 | -0.019 | 1 | 3 | 1 |
| M2 | OVERALL | jaccard | 54 | 1 | 0.561 | 0.602 | 0.042 | 2 | 3 | 1 |
| M2 | ACT | cpmr | 54 | 1 | 0.241 | 0.204 | -0.037 | 3 | 3 | 0 |
| M2 | MEANS | cpmr | 54 | 1 | 0.278 | 0.278 | 0.000 | 2 | 3 | 1 |
| M2 | PURPOSE | cpmr | 54 | 1 | 0.722 | 0.722 | 0.000 | 3 | 3 | 0 |
| M3 | OVERALL | macro_f1 | 54 | 1 | 0.568 | 0.650 | 0.082 | 2 | 1 | 1 |
| M3 | OVERALL | micro_f1 | 54 | 1 | 0.715 | 0.775 | 0.060 | 1 | 1 | 0 |
| M3 | OVERALL | exact_set | 54 | 1 | 0.000 | 0.056 | 0.056 | 2 | 1 | 1 |
| M3 | OVERALL | jaccard | 54 | 1 | 0.563 | 0.633 | 0.070 | 1 | 1 | 0 |
| M3 | ACT | cpmr | 54 | 1 | 0.426 | 0.556 | 0.130 | 2 | 2 | 0 |
| M3 | MEANS | cpmr | 54 | 1 | 0.444 | 0.537 | 0.093 | 1 | 2 | 1 |
| M3 | PURPOSE | cpmr | 54 | 1 | 0.741 | 0.796 | 0.056 | 2 | 2 | 0 |
| M4 | OVERALL | macro_f1 | 54 | 1 | 0.534 | 0.614 | 0.080 | 3 | 3 | 0 |
| M4 | OVERALL | micro_f1 | 54 | 1 | 0.702 | 0.765 | 0.064 | 3 | 2 | 1 |
| M4 | OVERALL | exact_set | 54 | 1 | 0.000 | 0.019 | 0.019 | 2 | 2 | 0 |
| M4 | OVERALL | jaccard | 54 | 1 | 0.542 | 0.615 | 0.072 | 3 | 2 | 1 |
| M4 | ACT | cpmr | 54 | 1 | 0.463 | 0.593 | 0.130 | 1 | 1 | 0 |
| M4 | MEANS | cpmr | 54 | 1 | 0.444 | 0.574 | 0.130 | 1 | 1 | 0 |
| M4 | PURPOSE | cpmr | 54 | 1 | 0.815 | 0.852 | 0.037 | 1 | 1 | 0 |

Family-level macro precision and recall deltas are:

| method | metric_scope | metric | dual_reference_n | excluded_incomplete_silver_reference_n | silver_reference_value | human_grounded_value | delta_human_minus_silver | silver_reference_dense_rank | human_grounded_dense_rank | rank_changed |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | ACT | macro_precision_family | 54 | 1 | 0.534 | 0.609 | 0.075 | 4 | 4 | 0 |
| M1 | ACT | macro_recall_family | 54 | 1 | 0.975 | 0.964 | -0.011 | 1 | 1 | 0 |
| M1 | MEANS | macro_precision_family | 54 | 1 | 0.329 | 0.353 | 0.024 | 4 | 4 | 0 |
| M1 | MEANS | macro_recall_family | 54 | 1 | 0.574 | 0.556 | -0.019 | 3 | 3 | 0 |
| M1 | PURPOSE | macro_precision_family | 54 | 1 | 0.259 | 0.259 | 0.000 | 4 | 4 | 0 |
| M1 | PURPOSE | macro_recall_family | 54 | 1 | 0.271 | 0.306 | 0.035 | 4 | 4 | 0 |
| M2 | ACT | macro_precision_family | 54 | 1 | 0.593 | 0.652 | 0.059 | 3 | 3 | 0 |
| M2 | ACT | macro_recall_family | 54 | 1 | 0.881 | 0.839 | -0.041 | 2 | 2 | 0 |
| M2 | MEANS | macro_precision_family | 54 | 1 | 0.441 | 0.479 | 0.038 | 3 | 3 | 0 |
| M2 | MEANS | macro_recall_family | 54 | 1 | 0.609 | 0.605 | -0.004 | 2 | 1 | 1 |
| M2 | PURPOSE | macro_precision_family | 54 | 1 | 0.621 | 0.718 | 0.097 | 1 | 1 | 0 |
| M2 | PURPOSE | macro_recall_family | 54 | 1 | 0.594 | 0.774 | 0.179 | 1 | 1 | 0 |
| M3 | ACT | macro_precision_family | 54 | 1 | 0.596 | 0.739 | 0.142 | 2 | 2 | 0 |
| M3 | ACT | macro_recall_family | 54 | 1 | 0.601 | 0.647 | 0.046 | 3 | 3 | 0 |
| M3 | MEANS | macro_precision_family | 54 | 1 | 0.642 | 0.617 | -0.025 | 1 | 1 | 0 |
| M3 | MEANS | macro_recall_family | 54 | 1 | 0.614 | 0.598 | -0.017 | 1 | 2 | 1 |
| M3 | PURPOSE | macro_precision_family | 54 | 1 | 0.543 | 0.668 | 0.125 | 3 | 2 | 1 |
| M3 | PURPOSE | macro_recall_family | 54 | 1 | 0.515 | 0.736 | 0.222 | 2 | 2 | 0 |
| M4 | ACT | macro_precision_family | 54 | 1 | 0.615 | 0.761 | 0.146 | 1 | 1 | 0 |
| M4 | ACT | macro_recall_family | 54 | 1 | 0.547 | 0.582 | 0.034 | 4 | 4 | 0 |
| M4 | MEANS | macro_precision_family | 54 | 1 | 0.542 | 0.614 | 0.072 | 2 | 2 | 0 |
| M4 | MEANS | macro_recall_family | 54 | 1 | 0.506 | 0.530 | 0.024 | 4 | 4 | 0 |
| M4 | PURPOSE | macro_precision_family | 54 | 1 | 0.564 | 0.647 | 0.083 | 2 | 3 | 1 |
| M4 | PURPOSE | macro_recall_family | 54 | 1 | 0.504 | 0.667 | 0.162 | 3 | 3 | 0 |

These deltas are descriptive and do not establish that either reference source is universally superior.

## 11. Auxiliary descriptive comparison

| target | substantive_n | comparable_n | exact_concordance | mean_jaccard |
|---|---|---|---|---|
| GEOGRAPHIC_FORM | 55 | 55 | 0.836 | 0.845 |
| MULTIPLICITY | 55 | 35 | 0.971 | N/A |
| CHILD | 55 | 14 | 1.000 | N/A |
| ORGANIZED_CRIMINAL_GROUP | 55 | 55 | 0.891 | N/A |

No auxiliary predictive model was trained or evaluated.

## 12. Limitations

- The reference was produced by one reviewer; reviewer-to-reviewer reliability is unavailable.
- Human labels are intentionally limited to the supplied narrative and need not reproduce broader SHERLOC metadata.
- The substantive and abstain samples are small; uncertainty intervals may be wide.
- Silver/human and jurisdiction-specific patterns are descriptive and should not be overinterpreted.
- Evaluation A was not modified, and no A4 auxiliary model benchmark was run.

## 13. Files

Canonical tables are under `outputs/analysis/evaluation_b/`; the four figures are under `outputs/figures/evaluation_b/`; the readable unexecuted view is `notebooks/10_human_grounded_evaluation.ipynb`; and this report is `docs/evaluation_b_human_grounded_report.md`.

## 14. Integrity

The immutable raw annotation source still matched SHA-256 `7ec0a40ab6a9d64588cf4b6c8b46d2572683cf7e340d786117604bc6f20081af`. Skip rows were excluded, and Abstain rows were retained only in the narrative-insufficiency diagnostic. No second reviewer was fabricated; reviewer-to-reviewer statistics were not computed. All retained human cases were excluded from supervised M1/M2 training, no active M4 demonstration overlapped the evaluated membership, and human labels were not used for model tuning. Evaluation A passed the preserved hash manifest before and after this analysis. No A4 auxiliary model benchmark was run.

Evaluation B is complete as a single-reviewer human-grounded narrative validation. The substantive human-reference subset and narrative-insufficiency subset were evaluated separately; no inter-annotator or adjudication analysis was performed, no retained human case entered supervised M1/M2 training, Evaluation A remained unchanged, and no A4 auxiliary model benchmark was run.