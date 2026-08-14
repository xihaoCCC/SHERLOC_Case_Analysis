# A2 jurisdiction-disjoint folds v1 report

Status: **PROVISIONAL — tied to `demo-bank-proposal-set-01-v1` pending human demo approval**  
Generator: `src/experiments/06_prepare_experiments.py` v1.1.0

## Evaluation universe and fold design

The frozen primary cohort contains exactly **18** jurisdiction/category values
with at least 20 cases: **861** cases total. The remaining **402** smaller-
jurisdiction cases can enter training/validation in every fold. The 18 values
were partitioned into three disjoint six-jurisdiction test groups by an
exhaustive deterministic balance objective. We first impose a near-equal test-
size constraint (range at most two cases), then minimize maximum absolute AMP-
prevalence deviation, summed squared deviation, size range, and lexicographic
group order. The generator recomputes this objective from the frozen benchmark
and fails if it no longer yields the recorded groups.

Test sizes are **288 / 287 / 286** (range 2). All 16 AMP labels observed in the
high-support universe appear in every test fold. All ten
`PURPOSE_REMOVAL_OF_ORGANS` cases occur in smaller jurisdictions, so the A2 test
count is unavoidably **0 / 0 / 0**; A2 cannot estimate jurisdiction-transfer
performance for that label.

### Fold 1

Held-out jurisdictions (6): Argentina; Australia; Republic of Moldova; Romania; Serbia; Slovakia

| Role | Cases |
|---|---:|
| TRAIN | 871 |
| VALIDATION | 98 |
| TEST | 288 |
| DEMO | 6 |
| Effective supervised training | 877 |

Membership SHA-256: `6b1b634024fe1800900b5835df0f35172765092b2d5f23ac9271e4a60c3325f2`. Validation splitter seed:
`20270820`.

| Label | TRAIN | VALIDATION | TEST | DEMO |
|---|---:|---:|---:|---:|
| `ACT_RECRUITMENT` | 702 | 84 | 235 | 4 |
| `ACT_TRANSPORTATION` | 538 | 61 | 222 | 4 |
| `ACT_TRANSFER` | 353 | 40 | 91 | 5 |
| `ACT_HARBOURING` | 439 | 49 | 116 | 4 |
| `ACT_RECEIPT` | 232 | 26 | 93 | 1 |
| `MEANS_THREAT_FORCE_OR_COERCION` | 470 | 53 | 138 | 3 |
| `MEANS_ABDUCTION` | 81 | 9 | 23 | 2 |
| `MEANS_FRAUD` | 216 | 24 | 63 | 1 |
| `MEANS_DECEPTION` | 432 | 49 | 190 | 2 |
| `MEANS_ABUSE_OF_POWER_OR_VULNERABILITY` | 495 | 62 | 209 | 3 |
| `MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL` | 87 | 10 | 42 | 2 |
| `PURPOSE_SEXUAL_EXPLOITATION` | 675 | 78 | 251 | 3 |
| `PURPOSE_FORCED_LABOUR_OR_SERVICES` | 190 | 21 | 37 | 1 |
| `PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES` | 47 | 5 | 11 | 1 |
| `PURPOSE_SERVITUDE` | 53 | 6 | 12 | 1 |
| `PURPOSE_REMOVAL_OF_ORGANS` | 8 | 1 | 0 | 1 |
| `PURPOSE_OTHER` | 48 | 5 | 12 | 1 |

### Fold 2

Held-out jurisdictions (6): Belgium; Brazil; Czechia; India; Philippines; Sweden

| Role | Cases |
|---|---:|
| TRAIN | 872 |
| VALIDATION | 98 |
| TEST | 287 |
| DEMO | 6 |
| Effective supervised training | 878 |

Membership SHA-256: `8e1464774d52ced1cbd394feb551f8e4fc83085973ad313527bff39f7a9a73cf`. Validation splitter seed:
`20280816`.

| Label | TRAIN | VALIDATION | TEST | DEMO |
|---|---:|---:|---:|---:|
| `ACT_RECRUITMENT` | 691 | 79 | 251 | 4 |
| `ACT_TRANSPORTATION` | 573 | 64 | 184 | 4 |
| `ACT_TRANSFER` | 325 | 37 | 122 | 5 |
| `ACT_HARBOURING` | 420 | 47 | 137 | 4 |
| `ACT_RECEIPT` | 249 | 28 | 74 | 1 |
| `MEANS_THREAT_FORCE_OR_COERCION` | 458 | 52 | 151 | 3 |
| `MEANS_ABDUCTION` | 85 | 10 | 18 | 2 |
| `MEANS_FRAUD` | 191 | 22 | 90 | 1 |
| `MEANS_DECEPTION` | 484 | 54 | 133 | 2 |
| `MEANS_ABUSE_OF_POWER_OR_VULNERABILITY` | 548 | 62 | 156 | 3 |
| `MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL` | 106 | 12 | 21 | 2 |
| `PURPOSE_SEXUAL_EXPLOITATION` | 676 | 76 | 252 | 3 |
| `PURPOSE_FORCED_LABOUR_OR_SERVICES` | 186 | 21 | 41 | 1 |
| `PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES` | 53 | 6 | 4 | 1 |
| `PURPOSE_SERVITUDE` | 58 | 6 | 7 | 1 |
| `PURPOSE_REMOVAL_OF_ORGANS` | 8 | 1 | 0 | 1 |
| `PURPOSE_OTHER` | 51 | 6 | 8 | 1 |

### Fold 3

Held-out jurisdictions (6): Canada; Colombia; Poland; Ukraine; United Kingdom of Great Britain and Northern Ireland; United States of America

| Role | Cases |
|---|---:|
| TRAIN | 873 |
| VALIDATION | 98 |
| TEST | 286 |
| DEMO | 6 |
| Effective supervised training | 879 |

Membership SHA-256: `3fb90a08c3fed6f15312a27437f4e8ec395e4c6687c1b346e4f70bc3cd94baae`. Validation splitter seed:
`20290827`.

| Label | TRAIN | VALIDATION | TEST | DEMO |
|---|---:|---:|---:|---:|
| `ACT_RECRUITMENT` | 716 | 79 | 226 | 4 |
| `ACT_TRANSPORTATION` | 566 | 64 | 191 | 4 |
| `ACT_TRANSFER` | 350 | 39 | 95 | 5 |
| `ACT_HARBOURING` | 409 | 46 | 149 | 4 |
| `ACT_RECEIPT` | 263 | 29 | 59 | 1 |
| `MEANS_THREAT_FORCE_OR_COERCION` | 432 | 49 | 180 | 3 |
| `MEANS_ABDUCTION` | 74 | 8 | 31 | 2 |
| `MEANS_FRAUD` | 208 | 23 | 72 | 1 |
| `MEANS_DECEPTION` | 480 | 44 | 147 | 2 |
| `MEANS_ABUSE_OF_POWER_OR_VULNERABILITY` | 510 | 57 | 199 | 3 |
| `MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL` | 103 | 12 | 24 | 2 |
| `PURPOSE_SEXUAL_EXPLOITATION` | 708 | 79 | 217 | 3 |
| `PURPOSE_FORCED_LABOUR_OR_SERVICES` | 152 | 17 | 79 | 1 |
| `PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES` | 37 | 4 | 22 | 1 |
| `PURPOSE_SERVITUDE` | 41 | 5 | 25 | 1 |
| `PURPOSE_REMOVAL_OF_ORGANS` | 8 | 1 | 0 | 1 |
| `PURPOSE_OTHER` | 49 | 6 | 10 | 1 |


## Leakage checks and use restriction

- Every high-support jurisdiction is TEST in exactly one fold.
- A held-out jurisdiction never appears in TRAIN, VALIDATION, or DEMO in its fold.
- The six proposed demos are outside all 18 high-support jurisdictions, have
  role DEMO in all folds, and are never scored.
- Each fold contains all 1,263 cases exactly once; the long file has 3,789 rows.
- Test labels are restricted to integrity checking until final evaluation and
  must not guide prompts, demos, hyperparameters, or thresholds.
- Demo replacement requires regenerating A2 train/validation assignments and
  hashes; held-out jurisdiction TEST membership remains independently fixed.
