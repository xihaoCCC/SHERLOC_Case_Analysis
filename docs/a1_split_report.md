# A1 IID split v1 report

Status: **PROVISIONAL — tied to `demo-bank-proposal-set-01-v1` pending human demo approval**  
Generator: `src/experiments/06_prepare_experiments.py` v1.1.0  
Seed family: `20260811`

## Exact allocation

| Role | Cases | Effective supervised training |
|---|---:|---:|
| TRAIN | 878 | 878 |
| VALIDATION | 126 | 0 |
| TEST | 253 | 0 |
| DEMO | 6 | 6 |

The six proposed demonstrations were reserved before splitting. The remaining
1,257 cases were divided by iterative multilabel stratification over all 17 AMP
indicators. Exact splitter seeds were `20261823` for TEST and
`20262821` for VALIDATION. M1/M2 effective training is
TRAIN + DEMO = **884**. M4 demonstrations are
excluded from every reported metric.

Membership SHA-256: `4360fe5100bee298ff3593446e554926f65bb06720a517e732219935962ee8fb`.

## AMP frequencies

| Label | TRAIN | VALIDATION | TEST | DEMO |
|---|---:|---:|---:|---:|
| `ACT_RECRUITMENT` | 722 | 95 | 204 | 4 |
| `ACT_TRANSPORTATION` | 574 | 82 | 165 | 4 |
| `ACT_TRANSFER` | 338 | 49 | 97 | 5 |
| `ACT_HARBOURING` | 422 | 60 | 122 | 4 |
| `ACT_RECEIPT` | 245 | 35 | 71 | 1 |
| `MEANS_THREAT_FORCE_OR_COERCION` | 462 | 66 | 133 | 3 |
| `MEANS_ABDUCTION` | 79 | 11 | 23 | 2 |
| `MEANS_FRAUD` | 212 | 30 | 61 | 1 |
| `MEANS_DECEPTION` | 471 | 65 | 135 | 2 |
| `MEANS_ABUSE_OF_POWER_OR_VULNERABILITY` | 535 | 77 | 154 | 3 |
| `MEANS_PAYMENTS_OR_BENEFITS_FOR_CONTROL` | 97 | 14 | 28 | 2 |
| `PURPOSE_SEXUAL_EXPLOITATION` | 704 | 101 | 199 | 3 |
| `PURPOSE_FORCED_LABOUR_OR_SERVICES` | 173 | 25 | 50 | 1 |
| `PURPOSE_SLAVERY_OR_SIMILAR_PRACTICES` | 44 | 6 | 13 | 1 |
| `PURPOSE_SERVITUDE` | 50 | 7 | 14 | 1 |
| `PURPOSE_REMOVAL_OF_ORGANS` | 6 | 1 | 2 | 1 |
| `PURPOSE_OTHER` | 45 | 7 | 13 | 1 |

The ten organ-removal cases allocate as TRAIN **6**,
VALIDATION **1**, TEST **2**,
and DEMO **1**. Validation and test therefore retain
the rare label.

## Geographic Form audit

| Role | N | Eligible | Internal | Transnational | Both |
|---|---:|---:|---:|---:|---:|
| TRAIN | 878 | 808 | 287 | 556 | 35 |
| VALIDATION | 126 | 116 | 44 | 76 | 4 |
| TEST | 253 | 226 | 99 | 136 | 9 |
| DEMO | 6 | 6 | 3 | 3 | 0 |

Form values were audited after AMP-first splitting. Ineligible Form cases are
not interpreted as two reference negatives.

## Integrity and use restriction

- Exactly 1,263 unique benchmark cases are assigned once.
- DEMO is disjoint from VALIDATION and TEST and is effective supervised training.
- No model result or test performance informed this split.
- A1 test labels may be used only for integrity checks until final evaluation.
- Prompt wording, demonstrations, hyperparameters, and thresholds must not be
  selected from A1 test errors or labels.
- If any proposed demo is rejected or replaced, regenerate this file and record
  a new membership hash before model execution.
