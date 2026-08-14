# Phase 4 Primary AMP Experiment Freeze v1

Status: **FINAL FROZEN BEFORE MODEL EXECUTION**  
Frozen: 2026-08-13  
Generator: `src/experiments/07_finalize_experiment_freeze.py` v1.0.0

This freeze was created without training, API calls, predictions, or model-result inspection. The primary reference is always described as the **SHERLOC Legacy-Keyword silver reference**.

## Frozen corpus and ontology

- Cohort ID: `sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`
- Cohort N: **1,263**
- Benchmark JSONL SHA-256: `2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020`
- Ontology: `sherloc-legacy-amp-v1` v1.0.0 (5 Acts, 6 Means, 6 Purposes; 17 outputs)
- Ontology SHA-256: `f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1`
- Primary input: exact English SHERLOC Fact Summary
- Primary target: exact Legacy SHERLOC Keyword AMP values mapped to the frozen ontology
- Geographic Form and all other auxiliary features are outside the primary AMP benchmark.

## Frozen demonstration bank

- Bank: `sherloc-amp-demo-bank-v1` v1.0.0
- Approved active six: `[1487, 1494, 1178, 498, 391, 157]`
- Approved reserve two: `[1343, 936]`
- Source human-review SHA-256: `c7e793e781c77bde4f99507b66b6ffeb5e37de768c86fd27f58c9e5cdf5e242f`
- Approved-case content hash: `3d533a53c9fffc4bd0f2a2d377319d384a404b6f4f93098760228bcd967c4b14`
- Aggregate bank-membership hash: `5905b0ef3533c268fa753098d64f9a625cd1a82c7a5c54bd7d2d9d0f61da14c7`
- Demo config file SHA-256: `1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b`
- Demonstration outputs contain AMP arrays only; they contain no auxiliary output.

| Setting | Ordered ranks | Demo/test jurisdiction overlap | Membership SHA-256 |
|---|---|---|---|
| A1 | 1487, 1494, 1178, 498, 391, 157 | none | `0e98d6196d4b7e1a3f15c81186a37e61e00ec34828f4b6bfb7c1398323f02eba` |
| A2_FOLD_1 | 1487, 1494, 1178, 498, 391, 157 | none | `e53b60b495899040eb8d51bfe7441203ea32939eaa13a911eafeb5df0c1dc3ec` |
| A2_FOLD_2 | 1487, 1494, 1178, 498, 157, 936 | none | `e7a06695dad296df8e21ec5c12653fe3a1c626946f429ee6c17e875ccb0e1453` |
| A2_FOLD_3 | 1487, 1494, 391, 157, 1343, 936 | none | `d983c920149a7fec28e7af2326f44508b6561bbf29d9441463e4f7f3be2b01d2` |

## Final A1 IID membership

- Counts: TRAIN=876, VALIDATION=126, TEST=253, ACTIVE_DEMO=6, RESERVE_DEMO=2
- Effective supervised training: **884** (TRAIN + ACTIVE_DEMO + RESERVE_DEMO)
- Iterative splitter seeds: TEST `20261813`, VALIDATION `20262822`
- Membership SHA-256: `edfecb1e885eee2e2418a4d26c053d90057555291bc494acc6896d881fed2ef8`
- CSV SHA-256: `63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d`
- All eight approved bank cases are outside validation/test. Only the active six are supplied to A1 M4.
- Organ-removal support: TRAIN=7, VALIDATION=1, TEST=2; the approved cases have no organ-removal label.

## Final A2 jurisdiction-disjoint membership

The verified >=20-case universe contains 18 jurisdictions and 861 cases. Every one is TEST in exactly one fold. The other 402 cases are never A2 TEST. Non-used approved cases follow their ordinary fold membership.

| Fold | Held-out jurisdictions | Counts | Validation seed | M4 ranks | Fold membership SHA-256 |
|---:|---|---|---:|---|---|
| 1 | Argentina; Australia; Republic of Moldova; Romania; Serbia; Slovakia | TRAIN=871, VALIDATION=98, TEST=288, ACTIVE_DEMO=6, RESERVE_DEMO=0 | 20270824 | 1487, 1494, 1178, 498, 391, 157 | `7be207cc8fd93dcce16a6d9e6091fb702c51163723f0ef759dcb674145250ee2` |
| 2 | Belgium; Brazil; Czechia; India; Philippines; Sweden | TRAIN=872, VALIDATION=98, TEST=287, ACTIVE_DEMO=5, RESERVE_DEMO=1 | 20280827 | 1487, 1494, 1178, 498, 157, 936 | `fd9ef8140716cce7def3b9d8b6977a8731211ef771bc893fd1fc456e6e80f5ac` |
| 3 | Canada; Colombia; Poland; Ukraine; United Kingdom of Great Britain and Northern Ireland; United States of America | TRAIN=873, VALIDATION=98, TEST=286, ACTIVE_DEMO=4, RESERVE_DEMO=2 | 20290834 | 1487, 1494, 391, 157, 1343, 936 | `c62d6778b466af37fabfc0ec0536569d1132c2eacec28849d4f8c82411b53579` |

- Aggregate A2 membership/fold hash: `0d45d408e1a6e4d4b513879485c5293465bea438fd7ff1479e20522e1cb00702`
- A2 CSV SHA-256: `75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67`
- Pooled held-out test N: **861** (288 + 287 + 286).
- All 10 organ-removal positives remain outside the A2 held-out universe; A2 TEST support is zero in every fold.
- Before every M4 fold, the runner must recheck that demo jurisdictions and held-out test jurisdictions are disjoint.

## Frozen prompts and model configurations

| Artifact | Version | SHA-256 |
|---|---|---|
| M1 config `config/experiments/m1_tfidf_logreg_amp_v2.yaml` | 2.0.0 | `44e80edf844d1589dec8b7236d58a65666f6479f0156d3c7ffff9e9de6d74b46` |
| M2 config `config/experiments/m2_modernbert_amp_v2.yaml` | 2.0.0 | `73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3` |
| LLM config `config/experiments/llm_extraction_amp_v2.yaml` | 2.0.0 | `5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b` |
| M3 prompt `prompts/m3_zero_shot_amp_v2.md` | m3-zero-shot-amp-v2 | `00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85` |
| M4 prompt `prompts/m4_six_shot_amp_v2.md` | m4-six-shot-amp-v2 | `2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37` |

Shared M3/M4 marked instruction block SHA-256: `b06d12d1efac3433ff0435ca589aa0aeec4328c5713e035bb7e029bc62468671`. The marked instructions are byte-identical; M4 adds only the six frozen solved message pairs.

Global random seed: `20260811`. M1 is TF-IDF + one-vs-rest logistic regression. M2 is one `answerdotai/ModernBERT-base` encoder with one 17-logit multilabel head. M3/M4 request `gpt-5.6-luna` through the Responses API with strict Structured Outputs, `store=false`, low reasoning, and low verbosity.

## Exact primary metric protocol

For A1, aggregate and per-label metrics use all 17 AMP dimensions. For A2, micro-F1, exact-set accuracy, and example Jaccard continue to use all 17 dimensions, so organ-removal false positives remain errors. Because pooled A2 reference support for `PURPOSE_REMOVAL_OF_ORGANS` is zero, its per-label precision/recall/F1 are reported **N/A**, not zero, and A2 macro-F1 is the unweighted mean over the other 16 labels with positive pooled reference support.

- Per-label precision = TP/(TP+FP), recall = TP/(TP+FN), and F1 is their harmonic mean; a zero denominator for a supported label yields 0.
- Macro-F1 is the arithmetic mean of eligible per-label F1 values. Micro-F1 pools TP/FP/FN across the stated dimensions.
- Exact-set accuracy is the proportion of cases whose full predicted and reference label sets match.
- Per-case Jaccard is intersection/union over all 17 labels; an empty prediction and empty reference score 1. The reported value is the case mean.
- M1/M2 predictions use one global validation-only threshold selected from 0.20, 0.25, ..., 0.80 by validation macro-F1. Ties choose the threshold closest to 0.50, then the smaller threshold. No per-label or test tuning is allowed. Threshold 0.50 is a secondary sensitivity result.
- Hyperparameter/checkpoint selection uses validation macro average precision only. Test labels never select preprocessing, hyperparameters, checkpoints, prompts, demonstrations, or thresholds.
- Confidence intervals use 1,000 deterministic case-level bootstrap resamples with seed `20260811` and percentile endpoints 2.5/97.5. A2 pooled bootstrap keeps the 16-label macro eligibility set fixed from the full pooled reference.
- A2 reports each fold, pooled OOD metrics over 861 unique test cases, and per-jurisdiction metrics. Distribution-shift deltas are pooled A2 minus A1; no significance claim is made without a designated interval/test.

## Test-set protection and scope

The A1/A2 test memberships, prompts, demonstration banks, ontology, primary metrics, and selection rules above must not be revised in response to model performance. Technical corrections are permitted only when independent of semantic test performance and must be documented before any protocol-preserving rerun.

Geographic Form, victim multiplicity, Sector, and child/minor involvement are explicitly secondary/exploratory. Any auxiliary evaluation must intersect its eligible cohort with these same A1/A2 memberships and must not alter or block the primary AMP benchmark.
