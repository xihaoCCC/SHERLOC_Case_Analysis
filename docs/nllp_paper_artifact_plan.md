# NLLP paper artifact plan

Status: planning scaffold only. This file deliberately contains no manuscript prose.

## Format constraint

- Target venue format: ACL/NLLP-style two-column paper.
- Main-text budget: 8 pages excluding references and permitted appendices.
- Keep numerical claims linked to `docs/paper_claim_to_evidence_map.md`.

## Eight-page allocation

| Main page budget | Section function | Primary artifact allocation |
|---|---|---|
| 0.75 | Problem framing and contributions | No table; define narrative extraction and the silver/human distinction |
| 0.75 | Related work and task positioning | No new result artifact |
| 1.00 | Corpus, ontology, and reference construction | Facts from `paper_methods_factsheet.md`; compact corpus/ontology text |
| 1.25 | M1-M4 methods and leakage controls | Method factsheet; no result table |
| 0.75 | A1/A2/A3 evaluation design and metrics | Split facts; CPMR definition; bootstrap protocol |
| 1.50 | Evaluation A results | `main_paper_results_table.csv`; Figure PF1; selective paired-CI statements |
| 1.25 | Human-grounded Evaluation A3 | `silver_vs_human_compact.csv`; Figures PF2/PF3; Abstain diagnostic |
| 0.75 | Auxiliary extension, limitations, and conclusion | `auxiliary_extension_compact.csv` only if space; single-reviewer and scope limitations |

## Main-paper artifacts

- `outputs/analysis/paper_final/main_paper_results_table.csv`
- `outputs/analysis/paper_final/silver_vs_human_compact.csv`
- `outputs/analysis/paper_final/model_behavior_summary.csv` (selected rows/metrics only)
- `outputs/analysis/paper_final/auxiliary_extension_compact.csv` (secondary compact result)
- Figure PF1: core A1/A2/A3 performance
- Figure PF2: Act/Means/Purpose CPMR
- Figure PF3: silver/human mismatch and reference-score shifts

Use no more than three paper-final figures. If space requires one removal, move PF2 to the appendix before removing PF1 or PF3.

## Appendix/supplement artifacts

- Full `master_results.csv` with CIs and Mean Contained Recall
- Full `paired_bootstrap_method_differences.csv` (63 rows)
- A1/A2 fold, jurisdiction, per-label, and rare-label tables
- A3 family/per-label, prediction-breadth, case-level, and Abstain case-level tables
- Full silver-versus-human case/per-label tables
- Auxiliary per-class and case-level tables
- Prompt, schema, demo-bank, execution, and cost/provenance details
- Reproducibility and pre-writing freeze manifests

## Claim discipline

- Use paired-CI language only for comparisons represented in the paired table.
- Do not convert `ci_excludes_zero` into an unplanned p-value or global significance claim.
- Keep CPMR explicitly secondary and descriptive.
- Use family-specific comparable N for silver-versus-human analyses.
- Describe A3 as a single-reviewer human-grounded narrative reference, not adjudicated gold.
- Describe Abstain outcomes as narrative-insufficiency diagnostics.
- Keep the auxiliary extension secondary: four zero-shot targets, no supervised baselines, no full-corpus silver benchmark.
- Do not claim that silver-only labels are erroneous.

## Pre-submission artifact checks

1. Validate `prewriting_freeze_manifest.json` against all cited files.
2. Generate every manuscript number from the frozen CSVs.
3. Cross-check every result sentence against the 13-row claim map.
4. Ensure figure captions identify reference type and N.
5. Confirm no Evaluation A/B canonical artifact changed after the pre-writing freeze.
6. Confirm public-data/privacy wording and `store=false` provenance.
