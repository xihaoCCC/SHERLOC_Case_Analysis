# Paper-final analysis package

This directory is a deterministic presentation layer over frozen canonical results. It does not recompute benchmark metrics and does not modify Evaluation A or Evaluation B artifacts.

## Tables

- `master_results.csv`: all A1/A2/A3 × M1-M4 core metrics, confidence intervals, family CPMR, and Mean Contained Recall.
- `main_paper_results_table.csv`: compact 12-row main-paper table.
- `silver_vs_human_compact.csv`: family-level dual-reference comparison with family-specific comparable N.
- `model_behavior_summary.csv`: A3 breadth, human-reference performance, CPMR, and Abstain behavior.
- `auxiliary_extension_compact.csv`: target-appropriate zero-shot auxiliary metrics.
- `paired_bootstrap_method_differences.csv`: separately generated canonical paired method-difference intervals; it is a required dependency and is never rewritten by the package builder.

## Figures

- `figure_pf1_core_performance.svg`: main paper; A1/A2/A3 Macro-F1, Micro-F1, and Jaccard.
- `figure_pf2_cpmr_by_family.svg`: main paper if space permits; otherwise appendix. CPMR is secondary.
- `figure_pf3_silver_human_reference_shift.svg`: main paper; silver/human family mismatch and A3 dual-reference score shifts.

## Main paper versus appendix

Use the compact main table, PF1, PF3, and selected PF2 panels in the main paper. Keep the full master table, all 63 paired-bootstrap rows, detailed family/per-label tables, case-level rows, and provenance manifests in the appendix or supplement.

## Reproduction and integrity

Run:

```bash
python src/experiments/25_build_paper_final_package.py --preflight
python src/experiments/25_build_paper_final_package.py --write
python src/experiments/25_build_paper_final_package.py --check
```

The builder validates every upstream dependency, Evaluation A's unchanged baseline, Evaluation B's analysis manifest, the auxiliary completion manifest, and the paired table before rendering. It refuses to overwrite any differing existing package file. `prewriting_freeze_manifest.json` is written last and hashes every generated artifact except itself.
