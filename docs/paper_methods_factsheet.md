# Paper methods factsheet

Status: deterministic pre-writing factual reference. This is not manuscript prose.

## Corpus and task

- Frozen SHERLOC trafficking-filter corpus: **1,590** cases.
- Cases with a usable English Fact Summary: **1,565**; without one: **25**.
- Primary AMP cohort: **1,263** cases with complete Legacy Keywords Act/Means/Purpose fields.
- Primary cohort ID: `sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`.
- Corpus manifest SHA-256: `1df2256dbfc063f88b23c2d85062f6650e91e49bd7a21cdb7a5fc11c94988fd5`.
- Primary benchmark JSONL SHA-256: `2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020`.
- Unit of analysis: one English Fact Summary per case.

## Frozen AMP ontology

- Act: **5** labels; Means: **6** labels; Purpose: **6** labels; total: **17**.
- Ontology ID/version: `sherloc-legacy-amp-v1` / `1.0.0`.
- Ontology SHA-256: `f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1`.
- Primary Evaluation A reference: SHERLOC Legacy Keywords silver reference.
- Evaluation A3 reference: single-reviewer human-grounded narrative reference.

## Evaluation designs

- **A1 IID:** TRAIN=876; VALIDATION=126; TEST=253; ACTIVE_DEMO=6; RESERVE_DEMO=2; split SHA-256 `63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d`.
- Effective A1 supervised training N=884 (TRAIN + ACTIVE_DEMO + RESERVE_DEMO); validation N=126 and TEST N=253 remained separate.
- **A2 jurisdiction-OOD:** 18 held-out jurisdictions; fold TEST Ns 288/287/286; pooled N=861; split SHA-256 `75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67`.
- A2 Fold 1 held out: Argentina, Australia, Republic of Moldova, Romania, Serbia, Slovakia.
- A2 Fold 2 held out: Belgium, Brazil, Czechia, India, Philippines, Sweden.
- A2 Fold 3 held out: Canada, Colombia, Poland, Ukraine, United Kingdom of Great Britain and Northern Ireland, United States of America.
- A2 official Macro-F1 uses the 16 labels with positive pooled reference support. `PURPOSE_REMOVAL_OF_ORGANS` remains a prediction dimension but has zero pooled support, receives per-label N/A, and is excluded from A2 Macro-F1.
- Organized Criminal Group (OCG) is an auxiliary feature and is irrelevant to A2, which evaluates AMP labels only.
- **A3 single reviewer:** source N=100; reviewed=74; Skip=13; retained=61; substantive=55; Abstain=6.
- Only one human reviewer was available for A3; no inter-annotator agreement, second-reviewer adjudication, or dual-reviewed gold reference was produced.
- A3 primary AMP scores use all 55 substantive cases. The six Abstain cases are analyzed separately and are not ordinary all-negative examples.

## M1-M4 fixed methods

### M1: TF-IDF plus one-vs-rest logistic regression

- Vectorizer: word 1-2 grams; lowercase; no accent stripping or stop-word list; token pattern `(?u)\b\w\w+\b`; L2 normalization; IDF with smoothing; sublinear TF; `min_df=2`; `max_df=1.0`; `max_features=50000`; float64.
- Classifier: one-vs-rest logistic regression; L2 penalty; liblinear solver; `C=1.0`; `class_weight=None`; `max_iter=2000`; tolerance `0.0001`; random seed `20260811`; `n_jobs=1` for the wrapper.
- One global validation-selected threshold: `0.25`; no per-label thresholds and no TEST-label tuning.
- Frozen config SHA-256: `44e80edf844d1589dec8b7236d58a65666f6479f0156d3c7ffff9e9de6d74b46`; A1 run-metadata SHA-256: `911e749ecdb4bfcdc5fef6a0339313cfc2f31f1a807044fcc5d9e404aff5264a`.
- A3 uses a dedicated leakage-free fit with training N=1,209 after excluding all 61 retained human-review cases; no A3 human labels were used for training or selection.

### M2: ModernBERT multilabel classifier

- Model/revision: `answerdotai/ModernBERT-base` / `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
- A1-selected settings: learning rate `3e-05`; weight decay `0.01`; fixed/selected duration `6` epochs; one global threshold `0.2`; no per-label thresholds or TEST-label tuning.
- Objective/optimization: unweighted `BCEWithLogitsLoss`; AdamW; linear learning-rate schedule with 0.1 warmup ratio; maximum gradient norm 1.0; training/data seed `20260811`.
- Tokenization/execution: max length `2048` with right truncation; Apple MPS device; physical train batch `1`; gradient accumulation `16`; effective batch `16`; BF16 autocast without gradient scaler; gradient checkpointing `true` with `use_reentrant=false`; AdamW `foreach=false`; dynamic right padding rounded to multiple 64; MPS low-watermark ratio 1.0 and no high-watermark override.
- Frozen config SHA-256: `73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3`; A1 run-metadata SHA-256: `711a26b26af783fa86f1b3f2a7c73f9a19f65ca63e0fcf3764786293dfcd0743`.
- A3 uses the fixed six-epoch transferred protocol with training N=1,209 after excluding all retained human-review cases; no A3 human labels were used for fitting, tuning, or epoch selection.

### M3/M4: OpenAI Responses API AMP extraction

- Model: `gpt-5.6-luna`; reasoning effort `low`; structured-output schema SHA-256 `d106c4ab1aa5bfcf34a6accd4f8c77df0bd21436cb0761d7828b21d9d87f46da`.
- M3: zero-shot; prompt SHA-256 `00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85`; prompt-file SHA-256 `00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85`.
- M4: six-shot; prompt SHA-256 `2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37`; prompt-file SHA-256 `2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37`.
- Demonstration bank: `sherloc-amp-demo-bank-v1`; active demonstrations=6; file SHA-256 `1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b`.
- A3 completion: M3 61/61; M4 61/61; retained-demo overlap=0.
- Each case was an independent request. `store=false`. The target payload contained only the public SHERLOC Fact Summary; no human-reference or SHERLOC silver labels were sent.
- Output-token policy: 512 initially; 2,048 only after explicit `incomplete/max_output_tokens` technical failures.
- Frozen LLM config SHA-256: `5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b`.

## Metrics and uncertainty

- Macro-F1: arithmetic mean of per-label F1 over the evaluation's supported-label set (17 labels in A1/A3; 16 in pooled A2).
- Micro-F1: pooled true-positive, false-positive, and false-negative decisions across cases and labels.
- Exact-set accuracy: proportion of cases whose complete predicted AMP set equals the reference set.
- Example-based Jaccard: mean case-level intersection-over-union of predicted and reference AMP sets.
- Family CPMR: proportion of cases with a nonempty predicted family set that is a subset of the reference family set. CPMR is a secondary descriptive diagnostic, not accuracy.
- Mean Contained Recall: mean predicted/reference set-size ratio among CPMR-successful cases only; N/A when there are no successes.
- Unpaired point-metric CIs: 1,000 case-level percentile-bootstrap resamples, seed 20260811.
- Paired method-difference CIs: 1000 paired case-level resamples, seed 20260811, 95% percentile intervals. `ci_excludes_zero` is descriptive; no unplanned p-value layer is added.

## Auxiliary extension

- Human-grounded zero-shot extension only; no supervised auxiliary baselines and no full-corpus silver auxiliary benchmark.
- Four targets over the substantive A3 source set: `GEOGRAPHIC_FORM` N=55, `VICTIM_MULTIPLICITY` N=55, `CHILD_INVOLVEMENT` N=55, `ORGANIZED_CRIMINAL_GROUP` N=55.
- Targets: Geographic Form, Victim Multiplicity, Child Involvement, and Organized Criminal Group.
- `UNKNOWN` is an evaluable class where defined. `Not Applicable` and explicit non-evaluable masks are excluded target-wise.
- Auxiliary completion-manifest SHA-256: `1ccd72a16b3109ade0593157a1081253bdfbaba6af2bb0d95144cc12ad7a167b`.

## Data handling and privacy boundary

- Model inputs were public SHERLOC English Fact Summaries. No private or nonpublic PII source was introduced.
- OpenAI requests used `store=false`; human annotations and SHERLOC silver-reference labels were excluded from target payloads.
- The benchmark preserves sensitive legal/trafficking narrative content locally; paper reporting should avoid unnecessary reproduction of identifying narrative detail.
