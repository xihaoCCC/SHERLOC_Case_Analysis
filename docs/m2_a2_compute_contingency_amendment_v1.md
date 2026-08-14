# M2 A2 compute-contingency protocol amendment v1

Date adopted: 2026-08-14

Protocol ID: `m2-modernbert-a2-fixed-a1-hparams-v1`

Affected experiment: M2 (ModernBERT), Evaluation A2 only.

## Reason for the amendment

The frozen Phase 4 implementation originally scheduled a fresh six-configuration
learning-rate/weight-decay search inside each of the three jurisdiction-disjoint
A2 folds. On the available Apple MPS hardware, one six-epoch configuration takes
approximately 43 minutes. Completing the original remaining work would therefore
require more than nine additional active-compute hours.

This amendment is a documented compute contingency. It was adopted before any
M2 A2 TEST prediction or TEST metric existed. No A2 TEST label, prediction, or
result informed the amendment.

## Previous protocol

For each A2 fold, train all six configurations in the frozen M2 grid, select a
configuration and epoch on that fold's VALIDATION data, select one global
threshold on VALIDATION data, and evaluate once on the held-out TEST
jurisdictions.

## Revised protocol

Transfer the hyperparameters selected independently by the completed M2 A1
VALIDATION procedure to every A2 fold without change:

- learning rate: `3e-5`
- weight decay: `0.01`
- sequence length: `2048`
- frozen-grid identity: `configuration_05`

Each A2 fold still uses a fresh initialization from
`answerdotai/ModernBERT-base` revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`. Each fold trains for at most six
epochs, selects its best checkpoint using that fold's VALIDATION macro average
precision, selects one global threshold using that fold's VALIDATION macro-F1
and the frozen threshold grid, and then evaluates exactly once on that fold's
held-out TEST cases.

The model family, 17-label head, input representation, maximum sequence length,
loss, effective batch size, gradient-checkpointing behavior, MPS bfloat16
autocast, optimizer implementation, padding policy, split membership, and
jurisdiction-disjoint evaluation design remain unchanged. The amendment removes
only the redundant per-fold hyperparameter searches.

## Fold 1 legacy-grid disposition

The interrupted original Fold 1 grid completed configurations 1 through 5.
Configuration 5 exactly matches the A1-transferred hyperparameters and passed a
strict artifact-integrity and split-leakage audit. It is therefore reused as the
official revised-protocol Fold 1 trained model; it is not selected by comparing
its Fold 1 validation score with configurations 1 through 4.

- Fold 1 configurations 1 through 4 remain historical, non-official artifacts.
- Fold 1 configuration 5 is reused without retraining.
- Fold 1 configuration 6 remains an interrupted, abandoned legacy-grid trial.
  It is neither resumed nor deleted.

## Unchanged experiments and safeguards

- M2 A1 and its selected model, threshold, predictions, and metrics remain
  unchanged.
- The frozen A1 TEST set and A2 jurisdiction folds remain unchanged.
- The AMP ontology, M3/M4 prompts, and demonstration bank remain unchanged.
- A2 TEST labels remain prohibited for epoch selection, threshold selection,
  preprocessing, hyperparameters, or any other design decision.
- M3, M4, API execution, and auxiliary-feature experiments are outside this
  amendment.

