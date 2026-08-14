# M2 A2 Fixed-A1-Hyperparameter Completion Report

Date: 2026-08-14  
Status: **COMPLETE**

## Protocol

M2 A2 used the documented compute-contingency protocol
`m2-modernbert-a2-fixed-a1-hparams-v1`. The A1-validation-selected
hyperparameters were transferred unchanged to every A2 fold:

- learning rate: `3e-5`
- weight decay: `0.01`
- maximum sequence length: `2048`

The amendment is recorded in
`docs/m2_a2_compute_contingency_amendment_v1.md` (SHA-256
`b83536e3b2cd8303f03b1977728c733e83a0599e1ed739e93846151ec29899ad`).
The decision was made for documented Apple MPS compute constraints before any
A2 TEST prediction or result was examined. A2 TEST labels were not used for
hyperparameter, epoch, threshold, preprocessing, or architecture selection.
M2 A1 was not rerun or changed.

Frozen inputs remained unchanged:

- A2 split SHA-256: `75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67`
- M2 config SHA-256: `73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3`
- ModernBERT revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`
- runner SHA-256: `fd4fc1041061ce814adb2a6e0e2264a502fb2dcc5c491954f8825ef829f37e3f`

## Fold execution

| Fold | Model action | Epochs completed | Best epoch | Validation macro AP | Validation macro-F1 at selected threshold | Threshold | TEST N | Fit/selection | Test inference | Total invocation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Existing validated legacy C5 reused; no retraining | 6 historical | 6 | 0.599506 | 0.537337 | 0.25 | 288 | 0.2 s | 44.9 s | 45.7 s |
| 2 | Fresh pinned pretrained initialization; only C5 trained | 6 | 6 | 0.607525 | 0.544645 | 0.20 | 287 | 42m56.9s | 41.3 s | 43m38.7s |
| 3 | Fresh pinned pretrained initialization; only C5 trained | 6 | 5 | 0.632702 | 0.553435 | 0.20 | 286 | 44m09.5s | 36.4 s | 44m46.6s |

Fold checkpoint SHA-256 values:

- Fold 1: `d4e81d832a9a43c1d34e6d8f7f46abd45898b6b906e88c2f42363c726f97c32b`
- Fold 2: `be117d223ab8f758c9dec78063c685bbd10bd206a89d389d9d50daad806530c8`
- Fold 3: `d9727e0d9550409e5e5d43a766cccf369e38f7ac3a9417e96138331a08bb0afc`

Fold 1 C1-C4 remain historical only. The interrupted legacy C6 remains
preserved and abandoned; its trial-state SHA-256 remains
`78cae1648b2d21714d7ac386ebc28842a69aa177b5c684883991bb3921e327e6`.

## Technical execution

The successful fixed-protocol executions used Apple MPS, physical training
batch size 1, evaluation batch size 2, gradient accumulation 16, effective
batch size 16, gradient checkpointing with `use_reentrant=false`, MPS BF16
autocast without a gradient scaler, `PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`,
AdamW `foreach=false`, right-side dynamic padding to a multiple of 64, and a
2048-token cap. Folds 2 and 3 each began from a fresh pinned pretrained
initialization; no weights were shared across folds. No OOM, batch fallback,
retry, or stall occurred.

A sandbox-hidden-MPS condition caused an initial Fold 1 inference-only attempt
to execute on CPU. Its metadata, fit derivatives, and predictions were
byte-preserved under
`outputs/models/m2/a2_fold_1/_protocol_history/m2-modernbert-a2-fixed-a1-hparams-v1/technical_reinference/`.
Only inference was regenerated on MPS; the official C5 checkpoint and legacy
grid were not modified or retrained. The archive event is linked from the
official Fold 1 run metadata.

## Canonical metrics

The shared evaluator processed the three explicit M2 prediction files with
1,000 case-level bootstrap resamples and seed `20260811`. Outputs are under
`outputs/metrics/stage5_m2_a2_validation/`.

Fold-level primary metrics:

| Fold | Macro-F1 | Micro-F1 | Exact-set accuracy | Example Jaccard | N |
|---|---:|---:|---:|---:|---:|
| 1 | 0.524329 | 0.721958 | 0.010417 | 0.564192 | 288 |
| 2 | 0.520183 | 0.695736 | 0.000000 | 0.534056 | 287 |
| 3 | 0.517220 | 0.699456 | 0.013986 | 0.540194 | 286 |

Pooled OOD primary metrics (`N=861`):

| Metric | Estimate | Bootstrap 95% CI |
|---|---:|---:|
| Macro-F1 (16 supported labels) | 0.527488 | [0.507216, 0.546926] |
| Micro-F1 | 0.705863 | [0.696875, 0.715396] |
| Exact-set accuracy | 0.008130 | [0.003484, 0.013966] |
| Example Jaccard | 0.546175 | [0.535853, 0.557338] |

`PURPOSE_REMOVAL_OF_ORGANS` has zero positive silver-reference support across
the pooled 861 A2 cases. It remains in all 17 prediction dimensions and in
micro/set metrics, has per-label F1 `N/A`, and is excluded only from the A2
macro-F1 average, which therefore uses 16 supported labels.

## Integrity validation

All completion checks passed:

- expected/observed/unique TEST cases were 288/288/288, 287/287/287, and
  286/286/286;
- pooled membership was 861 unique search ranks;
- all 18 held-out jurisdictions appeared as TEST in exactly one fold;
- no held-out jurisdiction appeared in that fold's TRAIN or VALIDATION rows;
- canonical URLs, jurisdictions, silver references, and case/search-rank
  identities matched the frozen split;
- each probability mapping contained the frozen 17 labels with finite values
  in `[0,1]`;
- the canonical evaluator validated the final A2 split and zero-support rule;
- all three run metadata records state that TEST labels were not used for
  selection;
- no run locks remained; and
- the full offline regression suite passed 117 of 117 tests.

Prediction SHA-256 values:

- Fold 1: `61fc3cf46e73e14b085fe3c63657f4a00d59a1d886c51a7d8f61a49158e9fb50`
- Fold 2: `4f86974ca8c804a567915d848f99f9f12ca6e8ea54b25178bd94d5b65c8d6847`
- Fold 3: `d4062740536aee28a478dda10edc67a034b967d92896a4f98712e0096ea3ef9f`

The measured MPS model-execution time for the completed fixed protocol was
approximately 1h29m11s (Fold 1 inference plus Fold 2 and Fold 3 full
invocations), consistent with the expected 1.5-2 hour window. The archived CPU
inference attempt added approximately 70 seconds. No M3/M4 runner was invoked,
no OpenAI API request was made, and no API-cost-bearing artifact was created.

M2 A2 is complete. The project is ready for the separate M3/M4 API task, which
must not begin automatically.
