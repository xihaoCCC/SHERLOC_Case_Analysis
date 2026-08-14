# Phase 4 execution report

Status: **IN PROGRESS**  
Execution date: 2026-08-13/14 (America/Chicago / UTC)

This report records technical execution events separately from the frozen
scientific protocol in `docs/experiment_freeze_v1.md`. No test result has been
used to alter a split, prompt, demonstration bank, target ontology, metric,
model family, hyperparameter grid, or threshold rule.

## Completed stages

- Stage 1: final freeze and integrity checks completed.
- Stage 2: M1 A1 completed.
- Stage 3: M1 A2 folds 1--3 completed.
- Reporting layer: canonical-output notebook generator and the three primary
  notebooks created; they remain gated until all M1--M4 outputs exist.

## M2 hardware-memory addendum

The first M2 A1 launch used the frozen 2,048-token input length and attempted
the required per-device batch fallback sequence 4, 2, and 1 while preserving a
nominal effective batch size of 16 through gradient accumulation. On Apple MPS
in full precision, each of those attempts exhausted available device memory
before completing an epoch. No validation selection, test inference, or M2
prediction artifact existed when those attempts ended.

The protocol-preserving hardware addendum therefore enables:

- MPS bfloat16 autocast (with no gradient scaler);
- gradient checkpointing with `use_reentrant=False`; and
- the same frozen 2,048-token limit, six-configuration validation grid,
  validation-only selection rule, seed, and nominal effective batch size.

The frozen configuration file was not changed; its SHA-256 remains
`73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3`.
The runner source loaded by the active A1 process has SHA-256
`052a950ccd7d5d6ddb0f7a1d5d99ae2464de6d6d07f0277bb2951166fefe1237`.

The addendum A1 launch attempted the complete frozen fallback sequence. All
three attempts failed with MPS out-of-memory while retaining
`max_length=2048`, before completing an epoch:

| Batch / accumulation | MPS allocated | Other allocations | MPS limit |
|---|---:|---:|---:|
| 4 / 4 | 9.22 GiB | 10.85 GiB | 20.13 GiB |
| 2 / 8 | 5.05 GiB | 15.04 GiB | 20.13 GiB |
| 1 / 16 | 2.76 GiB | 17.37 GiB | 20.13 GiB |

The monotonic increase in non-MPS-tracked allocations across successive
in-process fallbacks suggests that the failed Metal attempts were not fully
released by PyTorch cache cleanup. Before reducing the token limit, the next
technical check is therefore a new process that starts directly at batch 1
and accumulation 16, still at 2,048 tokens. Exactly one M2 process is
permitted per evaluation setting because the runner used for the failed
attempt did not yet provide an inter-process run lock.

That clean-process retry started at 2026-08-14 05:38:53 UTC. Its runner source
SHA-256 is
`a038e20ef483830f38c9360f89468e3a66c2831162a8b4cea1a034a3ecb1e796`.
The hardened runner now holds an atomic per-setting lock, archives the prior
failed artifacts and restart rationale under
`outputs/models/m2/_restart_history/`, and records the runtime/device context
in the resumability digest. The clean-process retry also failed before an
epoch, reporting 5.40 GiB MPS allocation plus 14.61 GiB other allocation at
the 20.13 GiB limit. This independently confirms that 2,048-token MPS training
is not feasible on the available machine even at batch 1 with bfloat16 and
gradient checkpointing.

The explicitly guarded 1,536-token contingency is therefore authorized for
the next run. It preserves complete inputs for 1,247/1,263 cases and records
16 truncated cases (compared with 1,254 complete and 9 truncated at 2,048).
No performance result informed this technical reduction.

The clean 1,536-token attempt also exhausted MPS memory before an epoch
(5.43 GiB MPS allocation plus 14.61 GiB other allocation). The final guarded
length contingency is therefore 1,024 tokens, which preserves complete inputs
for 1,214/1,263 cases and records 49 truncated cases. If that attempt also
fails, no further silent length reduction is permitted.

The 1,024-token attempt did fail before an epoch (4.43 GiB MPS allocation plus
15.58 GiB other allocation, followed by a 147.56 MiB request at the same
20.13 GiB hard limit). No shorter input will be attempted. Because the total
remains pinned near the allocator cap while the tensor allocation falls, the
next technical check lowers PyTorch's MPS *soft* watermark to request earlier
garbage collection/adaptive command-buffer commits. It does not raise or
disable the hard memory limit and will be recorded in the run context.

## M1 execution summary

M1 A1 selected `min_df=2`, `C=1.0`, no class weighting, and global threshold
0.25 using validation data only. M1 A2 used independently fitted fold models:

| Setting | Selected configuration | Threshold |
|---|---|---:|
| A2 Fold 1 | `min_df=2`, `C=4.0`, balanced class weights | 0.30 |
| A2 Fold 2 | `min_df=2`, `C=4.0`, no class weighting | 0.25 |
| A2 Fold 3 | `min_df=2`, `C=4.0`, no class weighting | 0.25 |

These values are execution records, not post-test protocol changes. Canonical
M1--M4 comparative tables will be generated only after the primary prediction
matrix is complete.

## Pending stages

- Finish M2 A1 and all three fresh A2 fine-tunes.
- Run M3/M4 authentication/schema dry runs on non-test cases.
- Run M3 then M4 for A1, generate A1 results, then run A2.
- Generate complete canonical A1/A2 metrics and distribution-shift tables.
- Execute and inspect the generated notebooks.
- Prepare auxiliary experiments only after the primary AMP benchmark succeeds.
