# SHERLOC AMP experiment reproducibility index

Status: Evaluation A complete; Evaluation B human reference frozen and final analysis gated on complete M1–M4 predictions.  
Repository snapshot inspected: 2026-08-15.  
Primary freeze: [`docs/experiment_freeze_v1.md`](experiment_freeze_v1.md).

This index identifies the authoritative artifacts and their roles. It does not
supersede the hashes, selection rules, or terminology in the experiment freeze.
Large SHERLOC-derived labels are **silver-reference labels**. Evaluation B uses
a **single-reviewer human-grounded narrative reference**; no second reviewer,
reviewer-to-reviewer statistic, or adjudication artifact is implied.

## Data and corpus artifacts

| Artifact | Role | Audit anchor |
|---|---|---|
| `data/manifests/case_urls.csv` | Frozen 1,590-record SHERLOC trafficking-filter membership | SHA-256 `1df2256dbfc063f88b23c2d85062f6650e91e49bd7a21cdb7a5fc11c94988fd5` |
| `logs/page_download_manifest.csv` | Requested/resolved URLs, raw filenames, response validation, byte hashes | `docs/page_download_report.md` |
| `data/interim/sherloc_cases_raw.jsonl` | Parser-v2 nested raw extraction | `docs/sherloc_extraction_contract_v2.md` and `logs/parser_diagnostics.json` |
| `outputs/metrics/parser_coverage.csv` | One-row-per-case structural parser coverage | `docs/parser_v2_report.md` |
| `data/processed/sherloc_benchmark_v1.jsonl` | Frozen 1,263-case primary benchmark | SHA-256 `2485b8f5aa9918a3e967e7d3602ec6005d99dd8f27a09a7c4306bbf193459020` |

The collection chain is URL manifest → unmodified raw HTML → download
manifest/provenance → parser-v2 JSONL → benchmark-v1. Corpus membership is the
frozen trafficking-filter result set, not the crime-type text embedded in a
case URL.

## Frozen splits

| Artifact | Design | SHA-256 |
|---|---|---|
| `data/splits/a1_iid_split_final_v1.csv` | A1 IID; TEST N=253 | `63a739fcb5a1d6af67a1ffc414f5b616a1e2ed7d063f7d34358ac7155803293d` |
| `data/splits/a2_jurisdiction_folds_final_v1.csv` | A2 jurisdiction-OOD; folds 288/287/286, pooled N=861 | `75ff2d87531bd9b68d2ee6382354d4191229eda4f3b3396d360349ad76e67f67` |

Generation and freeze validation live in
`src/experiments/06_prepare_experiments.py` and
`src/experiments/07_finalize_experiment_freeze.py`. Do not regenerate these
final split files for result reporting.

## Ontology

- `config/amp_ontology_v1.yaml`
- Ontology ID: `sherloc-legacy-amp-v1`
- 17 outputs: 5 Act, 6 Means, 6 Purpose
- SHA-256: `f01a61b5c27f5ed3cc7a8922ddf6ec5aa80f7fea487746d07be358050c5160c1`

## Model configurations

| Method | Frozen configuration | SHA-256 | Execution metadata |
|---|---|---|---|
| M1 | `config/experiments/m1_tfidf_logreg_amp_v2.yaml` | `44e80edf844d1589dec8b7236d58a65666f6479f0156d3c7ffff9e9de6d74b46` | `outputs/models/m1/*/run_metadata.json` |
| M2 | `config/experiments/m2_modernbert_amp_v2.yaml` | `73f5992afe934f1198f09382fb2ec38d0438831c157fc6ce44180798d51ba3e3` | `outputs/models/m2/*/run_metadata.json` |
| M3/M4 | `config/experiments/llm_extraction_amp_v2.yaml` | `5da03305ad97b36723c331ade7092147c828365abb32346b14a36726496d330b` | `outputs/logs/llm/*_diagnostics.json` and `outputs/metrics/{a1,a2}/amp_llm_api_usage.csv` |

M1 and M2 validation searches and selected-threshold artifacts are stored next
to their run metadata. M2 checkpoints/model weights are intentionally excluded
from version control; run metadata identifies the successful configuration.

## Prompts and demonstration banks

| Artifact | Role | SHA-256 |
|---|---|---|
| `prompts/m3_zero_shot_amp_v2.md` | Frozen M3 AMP prompt | `00b87b84356092b6d01b70f1a495f76c0ebd3ea49eb835a3bd7915a050a23f85` |
| `prompts/m4_six_shot_amp_v2.md` | Frozen M4 AMP prompt | `2d857b1a54b9ed2355558d5f1e8bc7dd3e216e37c5eb7397ffde8d82ee1bfb37` |
| `config/experiments/demo_bank_amp_v1.yaml` | Frozen active/reserve examples and fold-specific banks | `1f6316aa564e44222c5755843544244766daab7344dd002430f365aca235809b` |

The demo-bank source review is `data/annotations/demo_bank_review_v2.csv`.
Demo/test jurisdiction-disjointness and membership hashes are recorded in the
freeze document and the LLM prediction provenance.

## Technical amendments

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `docs/cpmr_metric_addendum_v1.md` | Secondary CPMR diagnostic frozen before A2 | `de72419d3d5e248742f244b1bb65719ffe4189b3b335f57dfc47b309544c492a` |
| `docs/llm_amp_technical_failure_amendment_v1.md` | Canonical label ordering and narrow 512→2048 completion fallback | `363c06abb49390a3cf66d646466313d6f50d655e41b801483063d1b180d7cb84` |
| `docs/m4_a2_rank_1340_technical_exception_addendum_v1.md` | Fold-1 rank-1340 rate-limit-only exception | `0ebb7945049d097476c3244407bff46b9f272704eb1a10118e649bfed2c8f6dc` |
| `docs/m2_hardware_execution_addendum_v1.md` | M2 hardware execution record | `182c3e9bfd857060c1efaa60fec629ed3bae891a37e1e5968e2a2c814d6e9f76` |
| `docs/m2_a2_compute_contingency_amendment_v1.md` | M2 A2 contingency protocol | `b83536e3b2cd8303f03b1977728c733e83a0599e1ed739e93846151ec29899ad` |
| `docs/m2_a2_fixed_protocol_completion_report.md` | M2 A2 completion record | `54cee6c4a30dc288407366221baf85ab19f0ddb7449180da49277ec7fe51819b` |

## Prediction artifacts

Canonical predictions are one JSONL file per method/evaluation scope:

- A1: `outputs/predictions/{m1,m2,m3,m4}/a1_test_predictions.jsonl`
- A2: `outputs/predictions/{m1,m2,m3,m4}/a2_fold_{1,2,3}_test_predictions.jsonl`

The canonical evaluator validates exact frozen membership and rejects missing,
duplicate, or extra rows. API retry/failure provenance is under
`outputs/logs/llm/`; model artifacts and logs are intentionally ignored by Git
because of size and operational detail.

## Canonical metrics and paper-facing analysis

Evaluator source: `src/experiments/11_evaluate_amp.py`. CPMR implementation:
`src/experiments/metrics.py`. API accounting:
`src/experiments/14_generate_a1_reporting_artifacts.py` and
`src/experiments/15_generate_a2_api_usage.py`.

- Completion manifest: `outputs/metrics/amp_evaluation_manifest.json`
- A1 canonical tables: `outputs/metrics/a1/`
- A2 canonical tables: `outputs/metrics/a2/`
- A1→A2 evaluator deltas: `outputs/metrics/amp_a1_to_a2_deltas.csv`
- Paper-facing deterministic tables: `outputs/analysis/evaluation_a/`
- Figure-label mapping: `outputs/analysis/evaluation_a/amp_label_display_mapping.csv`
- Four finalized SVG figures: `outputs/figures/evaluation_a/`
- Finalizer: `src/experiments/16_finalize_evaluation_a.py`
- Authoritative report: `docs/evaluation_a_final_report.md`
- Rare-label note: `docs/evaluation_a_rare_label_sensitivity.md`

Paper-facing tables and figures are views of existing frozen predictions and
canonical metrics. They do not create an alternative metric implementation.

## Evaluation B: single-reviewer human-grounded validation

- Annotation rules: `docs/human_annotation_guidelines_v1.md`
- Immutable annotation source: `data/annotations/reviewer_annotation_template.csv`
- Source manifest/QC: `outputs/analysis/evaluation_b/human_annotation_source_manifest.json`
  and `human_annotation_qc_summary.json`
- Frozen retained reference: `data/annotations/human_grounded_reference_v1.csv`
- Membership/leakage audit: `outputs/analysis/evaluation_b/eval_b_membership_manifest.json`
  and `eval_b_training_exclusion_audit.csv`
- Reference builder: `src/experiments/evaluation_b_reference.py`; CLI:
  `src/experiments/17_prepare_evaluation_b.py`
- Dedicated supervised runners: `src/experiments/20_run_evaluation_b_m1.py`
  and `21_run_evaluation_b_m2.py`
- Frozen-prompt LLM runner: `src/experiments/19_run_evaluation_b_llm.py`
- Read-only analysis/finalizer: `src/experiments/18_evaluate_evaluation_b.py`
- Pre-analysis Evaluation A integrity baseline:
  `outputs/analysis/evaluation_b/evaluation_a_integrity_baseline.json`

The finalizer fails closed unless the immutable annotation hash and QC chain,
retained membership, leakage audit, fixed execution metadata, and all M1–M4
prediction memberships validate. It performs no model training or API calls
and revalidates the Evaluation A hash baseline after writing Evaluation B.

Once the execution gate is complete, it writes canonical tables under
`outputs/analysis/evaluation_b/`, exactly four figures under
`outputs/figures/evaluation_b/`, and
`docs/evaluation_b_human_grounded_report.md`. These include the main/family/
per-label metrics, deterministic bootstrap intervals, nonempty- and
empty-reference CPMR diagnostics, Abstain diagnostics, prediction breadth,
silver-versus-human deltas, auxiliary descriptive concordance, and one
canonical case-level audit table. No auxiliary predictive benchmark is run.

## Notebooks

All notebooks below are deterministically generated by
`src/experiments/12_generate_analysis_notebooks.py`; generation does not execute
cells.

- `notebooks/07_a1_amp_results.ipynb`: finalized A1 tables and figures
- `notebooks/08_a2_amp_results.ipynb`: finalized pooled/fold/jurisdiction A2 views
- `notebooks/09_amp_error_analysis.ipynb`: canonical case-level inspection only
- `notebooks/10_human_grounded_evaluation.ipynb`: unexecuted thin reader of
  canonical Evaluation B QC, metrics, diagnostics, case rows, and four figures;
  missing artifacts display `NOT YET AVAILABLE`

Check reproducibility with:

```bash
python src/experiments/12_generate_analysis_notebooks.py --check
```

## Scripts

| Stage | Entry points |
|---|---|
| Collect/download/parse | `src/sherloc/01_collect_case_urls.py`, `02_download_pages.py`, `03_parse_pages.py` |
| Prepare/freeze | `src/experiments/06_prepare_experiments.py`, `07_finalize_experiment_freeze.py` |
| M1/M2/M3/M4 | `src/experiments/08_run_m1_tfidf.py`, `09_run_m2_modernbert.py`, `10_run_llm_amp.py` |
| Canonical evaluation | `src/experiments/11_evaluate_amp.py` |
| Paper analysis/notebooks | `src/experiments/16_finalize_evaluation_a.py`, `12_generate_analysis_notebooks.py` |
| Evaluation B reference/QC | `src/experiments/evaluation_b_reference.py`, `17_prepare_evaluation_b.py` |
| Evaluation B execution | `src/experiments/20_run_evaluation_b_m1.py`, `21_run_evaluation_b_m2.py`, `19_run_evaluation_b_llm.py` |
| Evaluation B analysis | `src/experiments/18_evaluate_evaluation_b.py`, `12_generate_analysis_notebooks.py` |

Running model entry points is not required to audit the preserved prediction
and metric artifacts.

## Tests

The offline suite is under `tests/`. Focused coverage includes freeze hashes,
M1/M2/LLM runners, request construction, canonical metrics and CPMR, API usage,
paper analysis, deterministic notebooks, parser-v2 behavior, and Evaluation B
utilities. Run without paid API calls:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python src/experiments/12_generate_analysis_notebooks.py --check
```

## Intentionally not versioned

The `.gitignore` policy excludes:

- secrets and local credentials: `api.txt`, `auth.json`, `.env`, `.env.*`;
- bulk raw HTML, interim parser output, and processed dataset copies;
- runtime logs;
- model/checkpoint directories and `*.safetensors` weights;
- notebook checkpoints, bytecode, local environments, and OS metadata.

These exclusions prevent credentials, bulky source material, and machine-local
runtime state from entering Git. Hashes, manifests, frozen configuration,
scripts, compact canonical tables/predictions, documentation, and tests provide
the reproducibility and audit trail. Large ignored artifacts require the
authorized local research archive or deterministic regeneration from the
preserved manifests; they should not be fetched from unverified sources.
