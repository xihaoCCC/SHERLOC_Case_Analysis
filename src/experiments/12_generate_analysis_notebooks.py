#!/usr/bin/env python3
"""Generate deterministic notebooks for finalized Evaluation A and Evaluation B.

Evaluation A notebooks load paper-facing tables and figures written by
``16_finalize_evaluation_a.py`` plus canonical case-level evaluator rows where
manual inspection is useful. They never reconstruct predictions or implement
metrics. The Evaluation B notebook is an unexecuted thin reader of finalized
single-reviewer analysis artifacts; unfinished stages report ``NOT YET
AVAILABLE`` without fabricating results.

The optional auxiliary notebook is retained only for backward-compatible
staging and is numbered 11 so it cannot collide with the human-grounded
Evaluation B notebook. Generating a notebook does not execute any cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


VERSION = "2.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks"

EVALUATION_A_NOTEBOOK_NAMES = (
    "07_a1_amp_results.ipynb",
    "08_a2_amp_results.ipynb",
    "09_amp_error_analysis.ipynb",
)
HUMAN_GROUNDED_NOTEBOOK_NAME = "10_human_grounded_evaluation.ipynb"
PRIMARY_NOTEBOOK_NAMES = EVALUATION_A_NOTEBOOK_NAMES
AUXILIARY_NOTEBOOK_NAME = "11_auxiliary_results.ipynb"

ANALYSIS_TABLES = (
    "a1_main_comparison.csv",
    "a2_main_comparison.csv",
    "amp_family_level_metrics.csv",
    "prediction_breadth_summary.csv",
    "rare_label_sensitivity.csv",
    "a1_to_a2_distribution_shift.csv",
    "m3_vs_m4_summary.csv",
    "m3_vs_m4_per_label_f1.csv",
    "amp_label_display_mapping.csv",
    "a2_fold_summary.csv",
    "a2_jurisdiction_summary.csv",
)
CORE_FIGURES = (
    "figure_1_a1_vs_a2_core_performance.svg",
    "figure_2_cpmr_by_amp_family.svg",
    "figure_3_cpmr_vs_contained_recall.svg",
    "figure_4_per_label_f1.svg",
)


def _cell_id(notebook_name: str, index: int, source: str) -> str:
    payload = f"{notebook_name}\0{index}\0{source}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _markdown(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _notebook(name: str, cells: Sequence[dict[str, Any]], *, purpose: str) -> dict[str, Any]:
    output_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        with_id = dict(cell)
        with_id["id"] = _cell_id(name, index, str(with_id["source"]))
        output_cells.append(with_id)
    return {
        "cells": output_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (xihao_env)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "sherloc_reporting": {
                "generator": "src/experiments/12_generate_analysis_notebooks.py",
                "generator_version": VERSION,
                "purpose": purpose,
                "evaluation_a_metric_source": "src/experiments/11_evaluate_amp.py",
                "evaluation_a_analysis_source": "src/experiments/16_finalize_evaluation_a.py",
                "evaluation_b_source": "src/experiments/evaluation_b.py",
                "cells_executed_by_generator": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = r'''from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, SVG, display


def locate_repo_root() -> Path:
    configured = os.environ.get("SHERLOC_REPO_ROOT")
    starts = [Path(configured).expanduser()] if configured else []
    starts.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in starts:
        if (candidate / "src/experiments/11_evaluate_amp.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate SHERLOC_Case_Analysis. Start Jupyter in the repository "
        "or set SHERLOC_REPO_ROOT."
    )


REPO_ROOT = locate_repo_root()
ANALYSIS_ROOT = REPO_ROOT / "outputs/analysis/evaluation_a"
FIGURE_ROOT = REPO_ROOT / "outputs/figures/evaluation_a"
METRICS_ROOT = REPO_ROOT / "outputs/metrics"


def load_csv(path: Path, required_columns=()) -> pd.DataFrame:
    """Load a finalized artifact without synthesizing missing rows."""
    if not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{path.relative_to(REPO_ROOT)}`"))
        return pd.DataFrame(columns=list(required_columns))
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def load_json(path: Path) -> dict:
    if not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{path.relative_to(REPO_ROOT)}`"))
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def show_table(frame: pd.DataFrame, *, empty_message="No finalized rows are available."):
    if frame.empty:
        display(Markdown(f"> **NOT YET AVAILABLE:** {empty_message}"))
    else:
        display(frame)


def show_figure(filename: str):
    """Display a finalized SVG; never recreate a figure in the notebook."""
    path = FIGURE_ROOT / filename
    if not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{path.relative_to(REPO_ROOT)}`"))
        return
    display(SVG(filename=str(path)))


manifest = load_json(METRICS_ROOT / "amp_evaluation_manifest.json")
completion_gate = manifest.get("final_completion_gate", "NOT YET AVAILABLE")
display(Markdown(f"**Canonical Evaluation A completion gate:** `{completion_gate}`"))
'''


def _artifact_inventory(table_names: Sequence[str], figure_names: Sequence[str]) -> str:
    return f'''table_names = {list(table_names)!r}
figure_names = {list(figure_names)!r}
artifact_inventory = pd.DataFrame([
    *[
        {{"artifact": str((ANALYSIS_ROOT / name).relative_to(REPO_ROOT)),
          "kind": "paper-facing table", "available": (ANALYSIS_ROOT / name).is_file()}}
        for name in table_names
    ],
    *[
        {{"artifact": str((FIGURE_ROOT / name).relative_to(REPO_ROOT)),
          "kind": "finalized figure", "available": (FIGURE_ROOT / name).is_file()}}
        for name in figure_names
    ],
])
display(artifact_inventory)
if not artifact_inventory["available"].all():
    display(Markdown("> **NOT YET AVAILABLE:** one or more finalized artifacts are missing."))
'''


def _a1_notebook() -> dict[str, Any]:
    name = EVALUATION_A_NOTEBOOK_NAMES[0]
    cells = [
        _markdown(
            """# Evaluation A1 — IID AMP Results

This notebook is the concise paper-facing view of finalized Evaluation A1. It loads canonical analysis artifacts produced from the frozen predictions; it does **not** implement metrics, bootstrap confidence intervals, sensitivity rules, or plots.

The reference labels are **SHERLOC silver-reference labels**, not ground truth. CPMR describes reference-contained prediction behavior and does not establish absolute factual correctness."""
        ),
        _markdown("## 1. Setup and finalized-artifact gate"),
        _code(COMMON_SETUP),
        _code(
            _artifact_inventory(
                (
                    "a1_main_comparison.csv",
                    "amp_family_level_metrics.csv",
                    "prediction_breadth_summary.csv",
                    "rare_label_sensitivity.csv",
                    "m3_vs_m4_summary.csv",
                    "m3_vs_m4_per_label_f1.csv",
                ),
                CORE_FIGURES[:2],
            )
        ),
        _markdown("## 2. Canonical A1 main comparison"),
        _code(
            r'''a1_main = load_csv(ANALYSIS_ROOT / "a1_main_comparison.csv", ("method", "n"))
show_table(a1_main)
'''
        ),
        _markdown(
            """## 3. Family-level performance and prediction breadth

These are finalized descriptive summaries. Prediction breadth is not a primary performance metric."""
        ),
        _code(
            r'''family_metrics = load_csv(
    ANALYSIS_ROOT / "amp_family_level_metrics.csv", ("evaluation", "method", "family")
)
prediction_breadth = load_csv(
    ANALYSIS_ROOT / "prediction_breadth_summary.csv", ("evaluation", "method")
)
show_table(family_metrics.loc[family_metrics["evaluation"].eq("A1")])
show_table(prediction_breadth.loc[prediction_breadth["evaluation"].eq("A1")])
'''
        ),
        _markdown(
            """## 4. Rare-label sensitivity analysis

This section is explicitly **descriptive sensitivity analysis**. It does not replace the frozen official Macro-F1."""
        ),
        _code(
            r'''rare_sensitivity = load_csv(
    ANALYSIS_ROOT / "rare_label_sensitivity.csv", ("evaluation", "method")
)
show_table(rare_sensitivity.loc[rare_sensitivity["evaluation"].eq("A1")])
'''
        ),
        _markdown("## 5. Descriptive M4 minus M3 comparison"),
        _code(
            r'''m3_m4 = load_csv(ANALYSIS_ROOT / "m3_vs_m4_summary.csv", ("evaluation",))
m3_m4_per_label = load_csv(
    ANALYSIS_ROOT / "m3_vs_m4_per_label_f1.csv", ("evaluation", "label_id")
)
show_table(m3_m4.loc[m3_m4["evaluation"].eq("A1")])
show_table(m3_m4_per_label.loc[m3_m4_per_label["evaluation"].eq("A1")])
display(Markdown("No statistical-significance claim is made for these descriptive differences."))
'''
        ),
        _markdown("## 6. Core figures"),
        _code(
            r'''show_figure("figure_1_a1_vs_a2_core_performance.svg")
show_figure("figure_2_cpmr_by_amp_family.svg")
'''
        ),
        _markdown(
            """## 7. Interpretation boundary

Use the finalized tables as the numeric source for paper writing. The figures are presentation views of those same artifacts. Do not tune any frozen model, threshold, prompt, demo bank, split, or ontology from this notebook."""
        ),
    ]
    return _notebook(name, cells, purpose="FINALIZED_EVALUATION_A1_VIEW")


def _a2_notebook() -> dict[str, Any]:
    name = EVALUATION_A_NOTEBOOK_NAMES[1]
    cells = [
        _markdown(
            """# Evaluation A2 — Jurisdiction-OOD AMP Results

This notebook loads the finalized pooled, fold, jurisdiction, shift, and sensitivity artifacts for the frozen A2 design. It does not recompute metrics or figures.

`PURPOSE_REMOVAL_OF_ORGANS` remains a prediction dimension but has zero positive A2 silver-reference support. Its per-label F1 is undefined where appropriate, and the official pooled A2 Macro-F1 uses the 16 supported labels. Jurisdiction rows are descriptive; small-N results must not be ranked as “best” or “worst.”"""
        ),
        _markdown("## 1. Setup and finalized-artifact gate"),
        _code(COMMON_SETUP),
        _code(
            _artifact_inventory(
                (
                    "a2_main_comparison.csv",
                    "amp_family_level_metrics.csv",
                    "prediction_breadth_summary.csv",
                    "rare_label_sensitivity.csv",
                    "a1_to_a2_distribution_shift.csv",
                    "m3_vs_m4_summary.csv",
                    "m3_vs_m4_per_label_f1.csv",
                    "amp_label_display_mapping.csv",
                    "a2_fold_summary.csv",
                    "a2_jurisdiction_summary.csv",
                ),
                CORE_FIGURES,
            )
        ),
        _markdown("## 2. Canonical pooled A2 comparison"),
        _code(
            r'''a2_main = load_csv(ANALYSIS_ROOT / "a2_main_comparison.csv", ("method", "n"))
show_table(a2_main)
'''
        ),
        _markdown("## 3. Family-level performance and prediction breadth"),
        _code(
            r'''family_metrics = load_csv(
    ANALYSIS_ROOT / "amp_family_level_metrics.csv", ("evaluation", "method", "family")
)
prediction_breadth = load_csv(
    ANALYSIS_ROOT / "prediction_breadth_summary.csv", ("evaluation", "method")
)
show_table(family_metrics.loc[family_metrics["evaluation"].eq("A2")])
show_table(prediction_breadth.loc[prediction_breadth["evaluation"].eq("A2")])
'''
        ),
        _markdown(
            """## 4. A1 to A2 distribution shift

Every delta is pooled A2 minus A1. These are descriptive differences; statistical significance was not tested."""
        ),
        _code(
            r'''shift = load_csv(
    ANALYSIS_ROOT / "a1_to_a2_distribution_shift.csv", ("method",)
)
show_table(shift)
'''
        ),
        _markdown("## 5. Descriptive M4 minus M3 comparison"),
        _code(
            r'''m3_m4 = load_csv(ANALYSIS_ROOT / "m3_vs_m4_summary.csv", ("evaluation",))
m3_m4_per_label = load_csv(
    ANALYSIS_ROOT / "m3_vs_m4_per_label_f1.csv", ("evaluation", "label_id")
)
show_table(m3_m4.loc[m3_m4["evaluation"].eq("A2")])
show_table(m3_m4_per_label.loc[m3_m4_per_label["evaluation"].eq("A2")])
'''
        ),
        _markdown("## 6. Fold and jurisdiction summaries"),
        _code(
            r'''fold_summary = load_csv(
    ANALYSIS_ROOT / "a2_fold_summary.csv", ("method", "fold", "n")
)
jurisdiction_summary = load_csv(
    ANALYSIS_ROOT / "a2_jurisdiction_summary.csv", ("jurisdiction", "method", "n")
)
show_table(fold_summary)
show_table(jurisdiction_summary)
display(Markdown(
    "Per-jurisdiction estimates are descriptive only. Do not rank jurisdictions or overinterpret small N."
))
'''
        ),
        _markdown("## 7. Rare-label sensitivity and support rule"),
        _code(
            r'''rare_sensitivity = load_csv(
    ANALYSIS_ROOT / "rare_label_sensitivity.csv", ("evaluation", "method")
)
show_table(rare_sensitivity.loc[rare_sensitivity["evaluation"].eq("A2")])
display(Markdown(
    "A2 Organ Removal support is zero. It remains in predictions and micro/set metrics but is already excluded from the official 16-supported-label Macro-F1."
))
'''
        ),
        _markdown("## 8. Recorded M3/M4 API execution summary"),
        _code(
            r'''a2_api_usage = load_csv(
    METRICS_ROOT / "a2/amp_llm_api_usage.csv", ("method", "scope")
)
show_table(a2_api_usage)
'''
        ),
        _markdown("## 9. Four core paper figures"),
        _code(
            r'''for figure_name in (
    "figure_1_a1_vs_a2_core_performance.svg",
    "figure_2_cpmr_by_amp_family.svg",
    "figure_3_cpmr_vs_contained_recall.svg",
    "figure_4_per_label_f1.svg",
):
    show_figure(figure_name)

label_display_mapping = load_csv(
    ANALYSIS_ROOT / "amp_label_display_mapping.csv",
    ("ontology_order", "label_id", "family", "display_label", "figure_short_label"),
)
display(Markdown("**Figure 4 short-label mapping to the full frozen ontology:**"))
show_table(label_display_mapping)
'''
        ),
    ]
    return _notebook(name, cells, purpose="FINALIZED_EVALUATION_A2_VIEW")


def _error_notebook() -> dict[str, Any]:
    name = EVALUATION_A_NOTEBOOK_NAMES[2]
    cells = [
        _markdown(
            """# Evaluation A — Canonical Error Inspection

This concise, post-hoc workspace filters the canonical one-row-per-case evaluator artifacts. It does not reconstruct predictions or recompute any metric. A mismatch with a SHERLOC silver-reference label is not automatically a factual error; later human adjudication is a separate Evaluation B activity."""
        ),
        _markdown("## 1. Setup"),
        _code(COMMON_SETUP),
        _markdown("## 2. Load canonical case-level rows"),
        _code(
            r'''error_columns = (
    "method", "case_id", "search_rank", "jurisdiction", "fold", "fact_summary",
    "silver_reference_amp_json", "predicted_amp_json", "false_positive_labels_json",
    "false_negative_labels_json", "exact_set_correct", "example_jaccard",
    "truncated_input", "act_cpmr", "act_contained_recall", "means_cpmr",
    "means_contained_recall", "purpose_cpmr", "purpose_contained_recall",
)
a1_errors = load_csv(METRICS_ROOT / "a1/amp_case_level_errors.csv", error_columns)
a2_errors = load_csv(METRICS_ROOT / "a2/amp_case_level_errors.csv", error_columns)
if not a1_errors.empty:
    a1_errors = a1_errors.assign(evaluation="A1")
if not a2_errors.empty:
    a2_errors = a2_errors.assign(evaluation="A2")
case_errors = pd.concat([a1_errors, a2_errors], ignore_index=True)
show_table(case_errors.head(20))
'''
        ),
        _markdown("## 3. Reproducible inspection filters"),
        _code(
            r'''# Record these values with every exported or cited case set.
EVALUATION = "A2"       # "A1", "A2", or None
METHOD = None           # "M1", "M2", "M3", "M4", or None
FOLD = None             # 1, 2, 3, or None
JURISDICTION = None     # exact string or None
LABEL_ID = None         # exact frozen ontology ID or None
EXACT_SET_FAILURES_ONLY = True
CPMR_FAMILY = None      # "act", "means", "purpose", or None
CPMR_SUCCESS_ONLY = False
MAX_ROWS = 100

filtered = case_errors.copy()
if EVALUATION is not None:
    filtered = filtered.loc[filtered["evaluation"].eq(EVALUATION)]
if METHOD is not None:
    filtered = filtered.loc[filtered["method"].eq(METHOD)]
if FOLD is not None:
    filtered = filtered.loc[filtered["fold"].eq(FOLD)]
if JURISDICTION is not None:
    filtered = filtered.loc[filtered["jurisdiction"].eq(JURISDICTION)]
if EXACT_SET_FAILURES_ONLY:
    filtered = filtered.loc[filtered["exact_set_correct"].eq(0)]
if LABEL_ID is not None:
    def contains_label(value):
        return LABEL_ID in json.loads(value)
    filtered = filtered.loc[
        filtered["false_positive_labels_json"].map(contains_label)
        | filtered["false_negative_labels_json"].map(contains_label)
    ]
if CPMR_FAMILY not in (None, "act", "means", "purpose"):
    raise ValueError("CPMR_FAMILY must be act, means, purpose, or None")
if CPMR_SUCCESS_ONLY:
    if CPMR_FAMILY is None:
        raise ValueError("Set CPMR_FAMILY before requesting CPMR successes")
    filtered = filtered.loc[filtered[f"{CPMR_FAMILY}_cpmr"].eq(1)]

inspection_columns = [
    "evaluation", "method", "search_rank", "jurisdiction", "fold",
    "exact_set_correct", "example_jaccard", "truncated_input",
    "silver_reference_amp_json", "predicted_amp_json",
    "false_positive_labels_json", "false_negative_labels_json",
    "act_cpmr", "act_contained_recall", "means_cpmr", "means_contained_recall",
    "purpose_cpmr", "purpose_contained_recall", "fact_summary",
]
show_table(filtered.sort_values(["evaluation", "search_rank"])[inspection_columns].head(MAX_ROWS))
'''
        ),
        _markdown("## 4. Finalized sensitivity and M3/M4 context"),
        _code(
            r'''rare_sensitivity = load_csv(
    ANALYSIS_ROOT / "rare_label_sensitivity.csv", ("evaluation", "method")
)
m3_m4_per_label = load_csv(
    ANALYSIS_ROOT / "m3_vs_m4_per_label_f1.csv", ("evaluation", "label_id")
)
show_table(rare_sensitivity)
show_table(m3_m4_per_label)
'''
        ),
        _markdown("## 5. Core diagnostic figures"),
        _code(
            r'''show_figure("figure_3_cpmr_vs_contained_recall.svg")
show_figure("figure_4_per_label_f1.svg")
'''
        ),
        _markdown(
            """## 6. Interpretation boundary

Any case selection developed after seeing benchmark outcomes is **post-hoc exploratory analysis** and must be reported as such. Do not use this notebook to tune or rerun Evaluation A, select human reliability cases, reveal hidden reviewer material, or adjudicate silver/human disagreements."""
        ),
    ]
    return _notebook(name, cells, purpose="CANONICAL_EVALUATION_A_ERROR_INSPECTION")


HUMAN_SETUP = r'''from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, SVG, display


def locate_repo_root() -> Path:
    configured = os.environ.get("SHERLOC_REPO_ROOT")
    starts = [Path(configured).expanduser()] if configured else []
    starts.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in starts:
        if (candidate / "src/experiments/18_evaluate_evaluation_b.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate SHERLOC_Case_Analysis. Start Jupyter in the repository "
        "or set SHERLOC_REPO_ROOT."
    )


REPO_ROOT = locate_repo_root()
ANALYSIS_ROOT = REPO_ROOT / "outputs/analysis/evaluation_b"
FIGURE_ROOT = REPO_ROOT / "outputs/figures/evaluation_b"
EXPECTED_ANALYSIS_OUTPUTS = {
    *(f"outputs/analysis/evaluation_b/{name}" for name in (
        "silver_vs_human_summary.csv",
        "silver_vs_human_per_label.csv",
        "silver_vs_human_case_level.csv",
        "auxiliary_silver_vs_human_summary.csv",
        "eval_b_main_results.csv",
        "eval_b_bootstrap_cis.csv",
        "eval_b_family_results.csv",
        "eval_b_per_label_results.csv",
        "eval_b_abstain_results.csv",
        "eval_b_abstain_case_level.csv",
        "eval_b_prediction_breadth.csv",
        "model_silver_vs_human_metric_comparison.csv",
        "human_grounded_case_level_errors.csv",
    )),
    *(f"outputs/figures/evaluation_b/{name}" for name in (
        "figure_b1_human_grounded_core_performance.svg",
        "figure_b2_human_grounded_cpmr.svg",
        "figure_b3_silver_vs_human_model_scores.svg",
        "figure_b4_silver_human_label_proportions.svg",
    )),
    "docs/evaluation_b_human_grounded_report.md",
}
REQUIRED_ANALYSIS_INPUTS = {
    "data/annotations/human_grounded_reference_v1.csv",
    "data/annotations/reliability_sample_100.csv",
    "data/processed/sherloc_benchmark_v1.csv",
    "config/experiments/demo_bank_amp_v1.yaml",
    "outputs/analysis/evaluation_b/eval_b_membership_manifest.json",
    "outputs/analysis/evaluation_b/human_grounded_reference_membership_v1.csv",
    "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv",
    "outputs/analysis/evaluation_b/human_annotation_source_manifest.json",
    "outputs/analysis/evaluation_b/human_annotation_qc_summary.json",
    "outputs/analysis/evaluation_b/evaluation_a_integrity_baseline.json",
    "outputs/models/evaluation_b/m1/run_metadata.json",
    "outputs/models/evaluation_b/m2/run_metadata.json",
    "outputs/logs/evaluation_b/llm/m3_diagnostics.json",
    "outputs/logs/evaluation_b/llm/m4_diagnostics.json",
    "outputs/predictions/evaluation_b/m1/predictions.jsonl",
    "outputs/predictions/evaluation_b/m2/predictions.jsonl",
    "outputs/predictions/evaluation_b/m3/eval_b_predictions.jsonl",
    "outputs/predictions/evaluation_b/m4/eval_b_predictions.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_analysis_manifest(manifest: dict) -> tuple[bool, str]:
    if not manifest:
        return False, "manifest missing"
    if manifest.get("status") != "COMPLETE":
        return False, "manifest status is not COMPLETE"
    inputs = manifest.get("inputs_sha256")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        return False, "manifest hash maps are missing"
    missing_inputs = REQUIRED_ANALYSIS_INPUTS - set(inputs)
    missing_outputs = EXPECTED_ANALYSIS_OUTPUTS - set(outputs)
    if missing_inputs or missing_outputs:
        return False, f"manifest is incomplete (inputs={sorted(missing_inputs)}, outputs={sorted(missing_outputs)})"
    for relative, expected in {**inputs, **outputs}.items():
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT) or not path.is_file():
            return False, f"bound artifact missing or outside repository: {relative}"
        if sha256_file(path) != str(expected):
            return False, f"bound artifact hash mismatch: {relative}"
    return True, "all frozen inputs and canonical outputs match"


def load_csv(relative: str, required_columns=()) -> pd.DataFrame:
    """Load a canonical evaluator table without synthesizing missing rows."""
    if relative in EXPECTED_ANALYSIS_OUTPUTS and not CANONICAL_ANALYSIS_READY:
        display(Markdown(f"> **NOT YET AVAILABLE:** canonical artifact gate failed for `{relative}`"))
        return pd.DataFrame(columns=list(required_columns))
    path = REPO_ROOT / relative
    if not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{relative}`"))
        return pd.DataFrame(columns=list(required_columns))
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{relative} is missing required columns: {sorted(missing)}")
    return frame


def load_json(relative: str) -> dict:
    path = REPO_ROOT / relative
    if not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{relative}`"))
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_inventory(method: str, relative: str, expected_n: int | None) -> dict:
    path = REPO_ROOT / relative
    if not path.is_file():
        return {"method": method, "path": relative, "present": False,
                "available": False, "expected_count": expected_n,
                "row_count": None, "validated_count": None}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    validated_count = sum(
        str(row.get("status", "")).upper() == "SUCCESS_VALIDATED" for row in rows
    )
    complete = expected_n is not None and len(rows) == expected_n == validated_count
    return {
        "method": method,
        "path": relative,
        "present": True,
        "available": complete,
        "expected_count": expected_n,
        "row_count": len(rows),
        "validated_count": validated_count,
    }


def show_table(frame: pd.DataFrame, *, message="canonical rows are unavailable"):
    if frame.empty:
        display(Markdown(f"> **NOT YET AVAILABLE:** {message}."))
    else:
        display(frame)


def show_figure(filename: str):
    path = FIGURE_ROOT / filename
    if not CANONICAL_ANALYSIS_READY or not path.is_file():
        display(Markdown(f"> **NOT YET AVAILABLE:** `{path.relative_to(REPO_ROOT)}`"))
        return
    display(SVG(filename=str(path)))


analysis_manifest = load_json(
    "outputs/analysis/evaluation_b/evaluation_b_analysis_manifest.json"
)
CANONICAL_ANALYSIS_READY, analysis_gate_detail = validate_analysis_manifest(analysis_manifest)
analysis_status = "COMPLETE" if CANONICAL_ANALYSIS_READY else "NOT YET AVAILABLE"
display(Markdown(
    f"**Canonical Evaluation B analysis:** `{analysis_status}` — {analysis_gate_detail}"
))
'''


def _human_notebook() -> dict[str, Any]:
    name = HUMAN_GROUNDED_NOTEBOOK_NAME
    cells = [
        _markdown(
            """# Evaluation B — Single-Reviewer Human-Grounded Narrative Validation

This unexecuted notebook is a thin, readable view of canonical Evaluation B artifacts. Missing upstream artifacts are reported as **NOT YET AVAILABLE**. Notebook cells do not construct the human reference, run models, call an API, implement metrics, bootstrap confidence intervals, or recreate figures.

Terminology: the reviewed labels form a **single-reviewer human-grounded narrative reference**. SHERLOC Legacy Keywords remain a separate **silver reference**. Abstain cases are narrative-insufficiency diagnostics, not ordinary all-negative references."""
        ),
        _markdown("## 1. Setup and canonical-artifact gate"),
        _code(HUMAN_SETUP),
        _markdown("## 2. Immutable annotation source and QC"),
        _code(
            r'''source_manifest = load_json(
    "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
)
qc_summary = load_json(
    "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
)
qc_report = load_csv(
    "outputs/analysis/evaluation_b/human_annotation_qc_report.csv"
)
if source_manifest and qc_summary:
    source = source_manifest.get("source", {})
    display(pd.DataFrame([{
        "immutable_source": source.get("path"),
        "source_rows": source.get("row_count", source_manifest.get("row_count")),
        "source_sha256": source.get("sha256", source_manifest.get("sha256")),
        "qc_status": qc_summary.get("status"),
        "reviewed_n": qc_summary.get("reviewed_n"),
        "not_reviewed_n": qc_summary.get("not_reviewed_n"),
        "skip_n": qc_summary.get("skip_n"),
        "substantive_n": qc_summary.get("substantive_n"),
        "abstain_n": qc_summary.get("abstain_n"),
        "retained_n": qc_summary.get("retained_n"),
        "blocking_issue_count": qc_summary.get("blocking_issue_count"),
    }]))
else:
    display(Markdown("> **NOT YET AVAILABLE:** source-manifest/QC summary pair."))
show_table(qc_report, message="human annotation QC issue rows are unavailable")
'''
        ),
        _markdown("## 3. Human AMP label supports"),
        _code(
            r'''per_label = load_csv(
    "outputs/analysis/evaluation_b/eval_b_per_label_results.csv",
    required_columns=("method", "family", "label_id", "support"),
)
if not per_label.empty:
    human_support = per_label.loc[
        per_label["method"].eq("M1"), ["family", "label_id", "support"]
    ].drop_duplicates()
    show_table(human_support, message="human AMP support table is unavailable")
else:
    display(Markdown("> **NOT YET AVAILABLE:** finalized human AMP supports."))
'''
        ),
        _markdown("## 4. Silver reference versus human narrative reference"),
        _code(
            r'''silver_summary = load_csv(
    "outputs/analysis/evaluation_b/silver_vs_human_summary.csv",
	required_columns=(
	    "family", "n", "substantive_n", "comparable_n",
	    "silver_reference_unavailable_n", "exact_set_concordance", "mean_jaccard",
	    "micro_precision_silver_against_human", "micro_recall_silver_against_human",
	    "micro_f1_silver_against_human", "shared_label_count",
	    "silver_only_label_count", "human_only_label_count",
	),
)
silver_per_label = load_csv(
	"outputs/analysis/evaluation_b/silver_vs_human_per_label.csv",
	required_columns=(
	    "family", "label_id", "n", "substantive_n", "comparable_n",
	    "silver_reference_unavailable_n", "silver_support", "human_support",
	    "shared", "silver_only", "human_only", "raw_agreement",
	),
)
auxiliary_summary = load_csv(
	"outputs/analysis/evaluation_b/auxiliary_silver_vs_human_summary.csv",
	required_columns=(
	    "target", "substantive_n", "comparable_n", "excluded_n",
	    "exact_concordance", "status",
	),
)
show_table(silver_summary, message="silver-versus-human family summary is unavailable")
show_table(silver_per_label, message="silver-versus-human per-label rows are unavailable")
show_table(auxiliary_summary, message="auxiliary descriptive comparison is unavailable")
'''
        ),
        _markdown("## 5. Leakage audit and prediction inventory"),
        _code(
            r'''leakage = load_csv(
    "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv",
    required_columns=(
        "reliability_case_id", "search_rank", "membership_sha256",
        "removed_from_eval_b_supervised_training",
        "removed_from_eval_b_validation", "removed_from_eval_b_threshold_tuning",
        "removed_from_eval_b_supervised_label_selection",
    ),
)
show_table(leakage, message="supervised leakage-exclusion audit is unavailable")

membership_manifest = load_json(
    "outputs/analysis/evaluation_b/eval_b_membership_manifest.json"
)
retained_n = membership_manifest.get("retained_n") if membership_manifest else None
overlap_n = (
    membership_manifest.get("a1_active_m4_demo_overlap_audit", {}).get("overlap_n")
    if membership_manifest else None
)
m4_expected_n = (
    int(retained_n) - int(overlap_n)
    if retained_n is not None and overlap_n is not None else None
)
prediction_inventory = pd.DataFrame([
    load_jsonl_inventory("M1", "outputs/predictions/evaluation_b/m1/predictions.jsonl", retained_n),
    load_jsonl_inventory("M2", "outputs/predictions/evaluation_b/m2/predictions.jsonl", retained_n),
    load_jsonl_inventory("M3", "outputs/predictions/evaluation_b/m3/eval_b_predictions.jsonl", retained_n),
    load_jsonl_inventory("M4", "outputs/predictions/evaluation_b/m4/eval_b_predictions.jsonl", m4_expected_n),
])
display(prediction_inventory)
if not prediction_inventory["available"].all():
    display(Markdown("> **NOT YET AVAILABLE:** one or more frozen prediction artifacts."))
'''
        ),
        _markdown(
            """## 6. Main M1–M4 human-grounded results

All primary rows come from one exact common substantive membership. The notebook displays canonical values and deterministic bootstrap intervals; it does not recompute them."""
        ),
        _code(
            r'''main_results = load_csv(
    "outputs/analysis/evaluation_b/eval_b_main_results.csv",
    required_columns=(
        "method", "n", "macro_f1", "micro_f1", "exact_set", "jaccard",
        "macro_supported_label_count", "act_cpmr", "means_cpmr", "purpose_cpmr",
    ),
)
bootstrap_cis = load_csv(
    "outputs/analysis/evaluation_b/eval_b_bootstrap_cis.csv",
    required_columns=("method", "metric", "estimate", "ci_low", "ci_high"),
)
show_table(main_results, message="main Evaluation B results are unavailable")
show_table(bootstrap_cis, message="bootstrap confidence intervals are unavailable")
'''
        ),
        _markdown("## 7. CPMR, contained recall, and empty-reference behavior"),
        _code(
            r'''family_results = load_csv(
    "outputs/analysis/evaluation_b/eval_b_family_results.csv",
    required_columns=(
        "method", "family", "cpmr", "mean_contained_recall",
        "macro_precision_family", "macro_recall_family", "macro_f1_family",
        "supported_label_count", "nonempty_reference_n",
        "cpmr_nonempty_reference", "empty_reference_n",
        "empty_reference_correct_empty_count", "empty_reference_correct_empty_rate",
    ),
)
show_table(family_results, message="family-level CPMR diagnostics are unavailable")
'''
        ),
        _markdown("## 8. Narrative-insufficiency (Abstain) diagnostic"),
        _code(
            r'''abstain_results = load_csv(
    "outputs/analysis/evaluation_b/eval_b_abstain_results.csv",
    required_columns=(
        "method", "abstain_n", "all_amp_empty_rate",
        "narrative_insufficiency_safe_rate", "mean_total_predicted_label_count",
        "cases_with_any_predicted_act", "cases_with_any_predicted_means",
        "cases_with_any_predicted_purpose",
    ),
)
abstain_cases = load_csv(
    "outputs/analysis/evaluation_b/eval_b_abstain_case_level.csv"
)
show_table(abstain_results, message="Abstain diagnostics are unavailable")
show_table(abstain_cases, message="Abstain case-level rows are unavailable")
'''
        ),
        _markdown("## 9. Prediction breadth and silver-scored versus human-scored behavior"),
        _code(
            r'''breadth = load_csv(
    "outputs/analysis/evaluation_b/eval_b_prediction_breadth.csv",
    required_columns=(
        "method", "n", "mean_predicted_act_labels", "mean_predicted_means_labels",
        "mean_predicted_purpose_labels", "mean_total_predicted_labels",
        "mean_total_human_labels", "silver_act_reference_available_n",
        "silver_means_reference_available_n", "silver_purpose_reference_available_n",
        "complete_silver_amp_reference_n", "mean_total_silver_labels",
    ),
)
reference_comparison = load_csv(
    "outputs/analysis/evaluation_b/model_silver_vs_human_metric_comparison.csv",
    required_columns=(
        "method", "metric_scope", "metric", "silver_reference_value",
        "human_grounded_value", "delta_human_minus_silver", "human_primary_n",
        "dual_reference_n", "excluded_incomplete_silver_reference_n",
    ),
)
show_table(breadth, message="prediction-breadth rows are unavailable")
show_table(reference_comparison, message="silver/human model-score deltas are unavailable")
'''
        ),
        _markdown("## 10. Core figures"),
        _code(
            r'''for figure_name in (
    "figure_b1_human_grounded_core_performance.svg",
    "figure_b2_human_grounded_cpmr.svg",
    "figure_b3_silver_vs_human_model_scores.svg",
    "figure_b4_silver_human_label_proportions.svg",
):
    show_figure(figure_name)
'''
        ),
        _markdown("## 11. Canonical case-level audit table"),
        _code(
            r'''case_level = load_csv(
    "outputs/analysis/evaluation_b/human_grounded_case_level_errors.csv",
    required_columns=(
        "reliability_case_id", "search_rank", "jurisdiction", "fact_summary",
        "human_act_json", "silver_act_json", "silver_act_reference_available",
        "complete_silver_amp_reference_available", "m1_prediction_json",
        "m2_prediction_json", "m3_prediction_json", "m4_prediction_json",
    ),
)
show_table(case_level, message="canonical case-level audit rows are unavailable")
'''
        ),
        _markdown(
            """## 12. Interpretation boundary

Only one human reviewer was available, so reviewer-to-reviewer reliability is unavailable. Silver-only labels are not automatically errors: SHERLOC structured metadata may be broader than information recoverable from a Fact Summary. Abstain results are descriptive insufficiency diagnostics. Small-N differences and auxiliary concordance must not be overinterpreted. Evaluation A remains frozen, and no auxiliary predictive benchmark is run here."""
        ),
    ]
    notebook = _notebook(name, cells, purpose="CANONICAL_EVALUATION_B_ANALYSIS_VIEW")
    notebook["metadata"]["sherloc_reporting"]["evaluation_b_analysis_source"] = (
        "src/experiments/18_evaluate_evaluation_b.py"
    )
    return notebook


def _auxiliary_notebook() -> dict[str, Any]:
    name = AUXILIARY_NOTEBOOK_NAME
    cells = [
        _markdown(
            """# Auxiliary Feature Results — Deferred

This optional notebook is separate from Evaluation A and Evaluation B. No Geographic Form, Multiplicity, Sector, or Child experiment is executed or reported here. It exists only as a deferred artifact inventory."""
        ),
        _markdown("## Status"),
        _code(
            r'''from IPython.display import Markdown, display
display(Markdown("**NOT YET AVAILABLE:** auxiliary experiments remain outside the current task."))
'''
        ),
    ]
    return _notebook(name, cells, purpose="DEFERRED_AUXILIARY_TEMPLATE")


def build_notebooks(*, include_auxiliary: bool = False) -> dict[str, dict[str, Any]]:
    """Return deterministic notebook documents keyed by output filename."""
    notebooks = {
        EVALUATION_A_NOTEBOOK_NAMES[0]: _a1_notebook(),
        EVALUATION_A_NOTEBOOK_NAMES[1]: _a2_notebook(),
        EVALUATION_A_NOTEBOOK_NAMES[2]: _error_notebook(),
        HUMAN_GROUNDED_NOTEBOOK_NAME: _human_notebook(),
    }
    if include_auxiliary:
        notebooks[AUXILIARY_NOTEBOOK_NAME] = _auxiliary_notebook()
    return notebooks


def canonical_notebook_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def write_notebooks(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    include_auxiliary: bool = False,
    check: bool = False,
) -> list[dict[str, Any]]:
    """Write or verify generated notebooks and return per-file diagnostics."""
    diagnostics: list[dict[str, Any]] = []
    for filename, notebook in build_notebooks(include_auxiliary=include_auxiliary).items():
        path = output_dir / filename
        expected = canonical_notebook_bytes(notebook)
        matches = path.is_file() and path.read_bytes() == expected
        if check and not matches:
            raise RuntimeError(f"Generated notebook is missing or stale: {path}")
        status = "UNCHANGED" if matches else "WOULD_WRITE" if check else "WRITTEN"
        if not check and not matches:
            _atomic_write(path, expected)
        diagnostics.append(
            {
                "path": str(path),
                "status": status,
                "sha256": hashlib.sha256(expected).hexdigest(),
                "cell_count": len(notebook["cells"]),
            }
        )
    return diagnostics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Also generate the deferred notebook 11 auxiliary inventory.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if a requested generated notebook is missing or stale.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        diagnostics = write_notebooks(
            args.output_dir,
            include_auxiliary=args.include_auxiliary,
            check=args.check,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps({"generator_version": VERSION, "notebooks": diagnostics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
