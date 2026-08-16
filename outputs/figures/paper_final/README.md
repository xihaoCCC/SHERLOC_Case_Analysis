# Paper-final figures

These three deterministic SVGs are presentation-only views of the frozen paper-final tables. They do not recompute benchmark metrics. Captions and manuscript text must identify the evaluation, reference type, and N from the cited table.

| Figure | Contents | Canonical table source | Recommended placement |
|---|---|---|---|
| `figure_pf1_core_performance.svg` | Macro-F1, Micro-F1, and example-based Jaccard for M1-M4 across A1, pooled A2, and A3 | `outputs/analysis/paper_final/master_results.csv` | Main results section; retain in the main paper |
| `figure_pf2_cpmr_by_family.svg` | Act, Means, and Purpose CPMR for M1-M4 across A1, pooled A2, and A3 | `outputs/analysis/paper_final/master_results.csv` | Secondary-behavior section; move to appendix first if space is tight |
| `figure_pf3_silver_human_reference_shift.svg` | Family-level silver/human mismatch and the change in A3 scores under human versus silver references | `outputs/analysis/paper_final/silver_vs_human_compact.csv` and the frozen Evaluation B dual-reference comparison | Human-grounded evaluation section; retain in the main paper |

## Reporting boundaries

- Use PF1 for core performance and PF3 for reference-source effects; CPMR in PF2 is a secondary descriptive diagnostic, not accuracy.
- A1/A2 use the SHERLOC Legacy Keywords silver reference. A3 uses a single-reviewer human-grounded narrative reference.
- Family-specific comparable N varies in the silver-versus-human panel; do not imply that silver-only labels are errors.
- Keep the package to these three paper-final figures. Detailed per-label, fold, jurisdiction, paired-bootstrap, and auxiliary plots belong in the appendix or supplement if later needed.
