# Evaluation A experiment-preparation report

Preparation date: `2026-08-11`  
Generator: `src/experiments/06_prepare_experiments.py` v1.1.0  
Frozen cohort: `sherloc-tip-2026-08-09-en-legacy-amp-complete-n1263-097ce2027171ebc9`  
Status: **PREPARATION COMPLETE; DEMO BANK AND DEPENDENT SPLITS PROVISIONAL**

No model was trained, no prediction was generated, and no OpenAI API request
was made in this stage.

## 1. A1 IID split

The six proposal-set-01 cases were reserved first. Iterative multilabel
stratification over all 17 AMP indicators then assigned the remaining 1,257
cases to TRAIN **878**, VALIDATION
**126**, and TEST
**253**; DEMO is **6**. M1/M2 effective training
is **884** because the six demos remain in
their supervised training pool. `PURPOSE_REMOVAL_OF_ORGANS` allocates
TRAIN/VALIDATION/TEST/DEMO = **6/1/2/1**. Membership SHA-256 is
`4360fe5100bee298ff3593446e554926f65bb06720a517e732219935962ee8fb`. All roles are disjoint and every
benchmark row appears once.

## 2. A2 jurisdiction-disjoint folds

The high-support universe is exactly **18 jurisdictions/categories and 861
cases**; 402 smaller-jurisdiction cases remain available to each non-test pool.
The exhaustive deterministic balance search produces:

| Fold | Held-out jurisdictions | TRAIN | VALIDATION | TEST | DEMO |
|---:|---|---:|---:|---:|---:|
| 1 | Argentina; Australia; Republic of Moldova; Romania; Serbia; Slovakia | 871 | 98 | 288 | 6 |
| 2 | Belgium; Brazil; Czechia; India; Philippines; Sweden | 872 | 98 | 287 | 6 |
| 3 | Canada; Colombia; Poland; Ukraine; United Kingdom of Great Britain and Northern Ireland; United States of America | 873 | 98 | 286 | 6 |

All 18 jurisdictions are held out exactly once, no held-out jurisdiction enters
its fold's TRAIN/VALIDATION/DEMO roles, and demos never enter TEST. All ten organ-
removal cases lie outside the high-support universe, making the A2 test count
unavoidably 0 in every fold.

## 3. Six-demonstration shortlist

Nine candidate banks were produced from frozen inputs using explicit eligibility
gates, ontology/Form coverage profiles, and a rare-label-weighted deterministic
set-cover search. Every proposal uses six distinct jurisdictions outside the A2
held-out universe and covers 5 Act, 6 Means, 6 Purpose, INTERNAL, and
TRANSNATIONAL reference values.

Proposal set 01 (the provisional split anchor) uses ranks
**31, 146, 955, 1293, 1494, 1517** from
**Malta; North Macedonia; Guatemala; Albania; Hungary; Jordan**.
It contains **628 words** (mean
**104.7**) and **758 ModernBERT summary tokens**. Its
reference coverage is **5/5 Acts,
6/6 Means, 6/6 Purposes**
and both Form values.

This is not the permanent M4 bank. The researcher/HT expert must review
`data/annotations/demo_bank_review.csv` and confirm that each Fact Summary
actually supports every Legacy reference label and Form value. Known concerns
include Slavery/Servitude, Transfer, Fraud, Internal Form, and focal-victim
status for the organ-removal example. Any replacement requires regeneration of
both split files and all membership hashes.

## 4. ModernBERT token audit

The official `answerdotai/ModernBERT-base` tokenizer was used at pinned revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`, with special tokens, across all 1,263 summaries.

- Min / median / mean: **22 / 234 / 339.1**
- P75 / P90 / P95 / P99: **423 / 659.6 / 921.6 / 1706.6**
- Maximum: **7,351**

| Max length | Fully covered | Percent | Truncated |
|---:|---:|---:|---:|
| 512 | 1,043 | 82.581% | 220 |
| 1,024 | 1,214 | 96.120% | 49 |
| 1,536 | 1,247 | 98.733% | 16 |
| 2,048 | 1,254 | 99.287% | 9 |
| 3,072 | 1,261 | 99.842% | 2 |
| 4,096 | 1,261 | 99.842% | 2 |
| 8,192 | 1,263 | 100.000% | 0 |

Recommended M2 `max_length`: **2,048**. It retains **1,254/1,263
(99.287%)** summaries without truncation
while avoiding the substantially higher compute burden of full 8,192-token
training. The nine truncated case identities remain visible in the audit.

## 5. Frozen M1/M2 plans

M1 uses summary-only word 1-2 gram TF-IDF (sublinear TF, at most 50,000 features)
with one-vs-rest L2 logistic regression. The small validation grid is
`min_df={1,2}`, `C={0.25,1,4}`, and `class_weight={null,balanced}`.

M2 uses one `answerdotai/ModernBERT-base` encoder and one 17-logit multilabel
head, standard BCE-with-logits, max length 2,048, effective batch size 16,
learning rates 1e-5/2e-5/3e-5, weight decay 0.01/0.05, at most six epochs,
patience 2, the model's default mean pooling and zero classifier dropout, and
BF16/FP16/FP32 runtime fallback.

For both methods, 0.5 is retained as a declared baseline. Model selection uses
validation macro average precision; one global threshold may then be selected
on validation macro-F1. Per-label thresholds and all test-label tuning are
disabled in v1. Geographic Form is an LLM-only auxiliary result in the current
freeze: M1/M2 remain the requested 17-output AMP models unless a separate
supervised Form head is preregistered later.

## 6. M3/M4 prompt and request contract

M3 `m3-zero-shot-v1` and M4 `m4-six-shot-v1` use the same byte-identical
instruction block, exact 5/6/6 ontology, AMP-plus-Form targets, target wrapper,
strict JSON Schema, `gpt-5.6-luna`, low reasoning effort, low verbosity, and
512 maximum output tokens. The schema has closed required objects, enum-limited
arrays with family-size limits, and two required Form booleans. Duplicate labels
are prohibited by instruction and rejected by host validation. No rationale,
chain of thought, confidence, evidence spans, multiplicity, child/minor, or
Sector is requested.

Luna is retained because the frozen design explicitly assigns the high-volume,
schema-constrained extraction role to that tier. This is a preregistered method
choice, not a claim that Luna will outperform other GPT-5.6 tiers; changing it
later would create a different experiment configuration.

The only substantive M4 difference is six frozen user/assistant demonstration
pairs inserted between the common developer instruction and target. The
preparation-only builder never imports an API client or sends a request and
fails closed until exactly six unique, ordered, expert-approved and frozen
demos outside the 18 A2 jurisdictions are explicitly supplied.

## 7. Offline input-token scale

These are planning estimates from the pinned ModernBERT tokenizer, not exact GPT
token counts, usage records, billing estimates, or API calls. The serialized
common instruction, schema, and empty target wrapper are approximately
**2,000 proxy tokens**.
The provisional six-demo messages add approximately
**1,404**.

- A1 TEST: **253 requests**; target median **250**,
  P90 **764.6**; M3 total approximately
  **597,842** input tokens and provisional-M4 total
  approximately **953,054**.
- A2: **861 M3 requests and 861 M4 requests**
  across three held-out folds; M3 approximately **2,003,671**
  input tokens and provisional-M4 approximately **3,212,515**.

No dollar cost is estimated because no versioned local price schedule is frozen.

## 8. Methodological and launch guards

- A1/A2 test labels are used now only for split-integrity checks. They must not
  guide prompts, demos, hyperparameters, checkpoints, or thresholds.
- Fit preprocessing, class weights, and supervised models on TRAIN + DEMO only;
  use VALIDATION for selection; never score DEMO.
- Form metrics, when run for M3/M4, use only the 1,156 eligible cases.
- Preserve raw model/API outputs later; classify refusals, incomplete responses,
  API errors, schema errors, and duplicate-label errors explicitly. Never repair
  or silently coerce them into empty labels.

Before execution: (1) approve and freeze six demonstrations and rerun this
generator; (2) decide whether the model alias should be replaced by an available
dated snapshot and record the returned model identifier; (3) configure the API
key in the execution environment, not the repository; and (4) explicitly decide
whether Form remains LLM-only auxiliary or receives a separately preregistered
supervised comparator.

## 9. Reproducibility and validation

Random seed family: `20260811`. Frozen benchmark and ontology hashes are checked at
startup. The generator also fails closed on Python, NumPy, scikit-learn,
iterative-stratification, Transformers, or tokenizers version drift. A2 groups
are independently recomputed. The builder records input,
prompt, schema, demonstration-bank, config, and canonical request-payload hashes.
The experiment, request-builder, parser-v2, and benchmark safety tests must pass
before model execution.
