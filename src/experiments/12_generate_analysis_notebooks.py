#!/usr/bin/env python3
"""Generate thin, reproducible Phase-4 AMP analysis notebooks.

The generated notebooks are presentation and inspection clients for the
canonical artifacts written by ``11_evaluate_amp.py``.  They deliberately do
not import metric implementations, reconstruct predictions, bootstrap cases,
or calculate F1/Jaccard from labels.  Missing canonical artifacts are reported
as pending so the notebooks remain usable while the staged experiment is in
progress, without presenting partial results as final.

By default this script creates the three primary AMP notebooks.  The optional
auxiliary notebook is only generated with ``--include-auxiliary`` because the
frozen protocol places auxiliary work after completion of the primary AMP
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence


VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "notebooks"

PRIMARY_NOTEBOOK_NAMES = (
    "07_a1_amp_results.ipynb",
    "08_a2_amp_results.ipynb",
    "09_amp_error_analysis.ipynb",
)
AUXILIARY_NOTEBOOK_NAME = "10_auxiliary_results.ipynb"


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


def _notebook(name: str, cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
                "metric_source": "src/experiments/11_evaluate_amp.py",
                "canonical_metrics_root": "outputs/metrics",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = r'''from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

EXPECTED_METHODS = ("M1", "M2", "M3", "M4")


def locate_repo_root() -> Path:
    """Locate the repository without relying on the notebook launch directory."""
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
METRICS_ROOT = REPO_ROOT / "outputs/metrics"


def load_json(path: Path) -> dict:
    if not path.is_file():
        display(Markdown(f"> **Pending:** `{path.relative_to(REPO_ROOT)}` does not exist."))
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path, required_columns=()) -> pd.DataFrame:
    """Load an evaluator table, reporting absence without synthesizing results."""
    if not path.is_file():
        display(Markdown(f"> **Pending:** `{path.relative_to(REPO_ROOT)}` does not exist."))
        return pd.DataFrame(columns=list(required_columns))
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing canonical columns: {sorted(missing)}")
    return frame


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def show_or_pending(frame: pd.DataFrame, message="No canonical rows are available yet."):
    if frame.empty:
        display(Markdown(f"> **Pending:** {message}"))
    else:
        display(frame)


def methods_complete(manifest: dict, evaluation: str) -> bool:
    methods = manifest.get("evaluations", {}).get(evaluation, {}).get("methods", [])
    return set(methods) == set(EXPECTED_METHODS)


manifest_path = METRICS_ROOT / "amp_evaluation_manifest.json"
evaluation_manifest = load_json(manifest_path)
'''


AVAILABILITY_CELL = r'''canonical_inputs = [
    METRICS_ROOT / "amp_evaluation_manifest.json",
    METRICS_ROOT / "a1/amp_primary_results.csv",
    METRICS_ROOT / "a1/amp_per_label.csv",
    METRICS_ROOT / "a1/amp_bootstrap_cis.csv",
    METRICS_ROOT / "a1/amp_case_level_errors.csv",
    METRICS_ROOT / "a2/amp_primary_results.csv",
    METRICS_ROOT / "a2/amp_per_fold.csv",
    METRICS_ROOT / "a2/amp_per_label.csv",
    METRICS_ROOT / "a2/amp_per_jurisdiction.csv",
    METRICS_ROOT / "a2/amp_bootstrap_cis.csv",
    METRICS_ROOT / "a2/amp_case_level_errors.csv",
    METRICS_ROOT / "amp_a1_to_a2_deltas.csv",
]
availability = pd.DataFrame(
    {
        "artifact": [str(path.relative_to(REPO_ROOT)) for path in canonical_inputs],
        "available": [path.is_file() for path in canonical_inputs],
    }
)
display(availability)

gate = evaluation_manifest.get("final_completion_gate", "PENDING")
complete = (
    gate == "PASSED_M1_M2_M3_M4_A1_A2"
    and methods_complete(evaluation_manifest, "A1")
    and methods_complete(evaluation_manifest, "A2")
)
if complete:
    display(Markdown("**Canonical completion gate: PASSED for M1-M4 in A1 and A2.**"))
else:
    display(Markdown(
        "> **Incomplete benchmark:** canonical M1-M4 A1/A2 outputs are not complete. "
        "Any available rows are technical previews, not the final comparison."
    ))
'''


def _a1_notebook() -> dict[str, Any]:
    name = PRIMARY_NOTEBOOK_NAMES[0]
    cells = [
        _markdown(
            """# A1 AMP Results — IID Test Set

This notebook is a thin presentation layer for the **canonical evaluator outputs** produced by `src/experiments/11_evaluate_amp.py`. It does not calculate F1, Jaccard, bootstrap intervals, thresholds, or predictions. The reference is the **SHERLOC Legacy Keywords silver reference**, not human-adjudicated gold labels.

The notebook reports incomplete artifacts as pending. It does not generate scientific conclusions or change the frozen protocol."""
        ),
        _markdown("## 1. Setup and canonical-artifact contract"),
        _code(COMMON_SETUP),
        _markdown("## 2. Artifact availability and completion gate"),
        _code(AVAILABILITY_CELL),
        _markdown("## 3. Frozen experiment metadata"),
        _code(
            r'''ontology_path = REPO_ROOT / "config/amp_ontology_v1.yaml"
demo_path = REPO_ROOT / "config/experiments/demo_bank_amp_v1.yaml"
llm_path = REPO_ROOT / "config/experiments/llm_extraction_amp_v2.yaml"
m1_config_path = REPO_ROOT / "config/experiments/m1_tfidf_logreg_amp_v2.yaml"
m2_config_path = REPO_ROOT / "config/experiments/m2_modernbert_amp_v2.yaml"

ontology = load_json(ontology_path)
demo_bank = load_json(demo_path)
llm_config = load_json(llm_path)
m1_config = load_json(m1_config_path)
m2_config = load_json(m2_config_path)

frozen_metadata = pd.DataFrame([
    {"item": "Primary cohort", "value": m1_config.get("primary_cohort_id", "PENDING")},
    {"item": "Reference terminology", "value": evaluation_manifest.get("reference_terminology", "PENDING")},
    {"item": "Ontology", "value": ontology.get("ontology_id", "PENDING")},
    {"item": "Ontology SHA-256", "value": sha256_file(ontology_path) if ontology_path.is_file() else "PENDING"},
    {"item": "Demo bank", "value": demo_bank.get("bank_id", "PENDING")},
    {"item": "Demo-bank SHA-256", "value": sha256_file(demo_path) if demo_path.is_file() else "PENDING"},
    {"item": "M3 prompt SHA-256", "value": llm_config.get("methods", {}).get("M3", {}).get("prompt_sha256", "PENDING")},
    {"item": "M4 prompt SHA-256", "value": llm_config.get("methods", {}).get("M4", {}).get("prompt_sha256", "PENDING")},
    {"item": "Bootstrap protocol", "value": json.dumps(evaluation_manifest.get("bootstrap", {}), sort_keys=True)},
])
display(frozen_metadata)
'''
        ),
        _markdown("## 4. A1 split composition"),
        _code(
            r'''a1_split_path = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
a1_split = load_csv(a1_split_path, ("search_rank", "split", "effective_supervised_train"))
if not a1_split.empty:
    split_composition = (
        a1_split.groupby("split", dropna=False)
        .agg(cases=("search_rank", "size"), supervised_train=("effective_supervised_train", "sum"))
        .reset_index()
    )
    display(split_composition)
    display(pd.DataFrame([{"split_sha256": sha256_file(a1_split_path)}]))
else:
    show_or_pending(a1_split)
'''
        ),
        _markdown("## 5. Frozen silver-reference label distribution by A1 role"),
        _code(
            r'''label_ids = [
    item["id"]
    for family in ("ACT", "MEANS", "PURPOSE")
    for item in ontology.get("families", {}).get(family, [])
]
if not a1_split.empty and label_ids:
    missing_labels = set(label_ids) - set(a1_split.columns)
    if missing_labels:
        raise ValueError(f"A1 split lacks ontology columns: {sorted(missing_labels)}")
    label_distribution = a1_split.groupby("split")[label_ids].sum().T
    label_distribution.index.name = "label_id"
    display(label_distribution)
else:
    display(Markdown("> **Pending:** frozen split or ontology is unavailable."))
'''
        ),
        _markdown("## 6. M1 and M2 frozen configurations and validation-selected thresholds"),
        _code(
            r'''def model_run_summary(method: str) -> dict:
    metadata = load_json(REPO_ROOT / f"outputs/models/{method.lower()}/a1/run_metadata.json")
    selection = metadata.get("selection", {})
    return {
        "method": method,
        "status": metadata.get("status", "PENDING"),
        "run_id": metadata.get("run_id", "PENDING"),
        "selected_global_threshold": selection.get("selected_global_threshold", "PENDING"),
        "selected_hyperparameters": json.dumps(selection.get("selected_hyperparameters", {}), sort_keys=True),
        "selection_data": "VALIDATION_ONLY",
        "test_labels_used_for_selection": metadata.get("test_labels_used_for_selection", "PENDING"),
    }

display(pd.DataFrame([model_run_summary("M1"), model_run_summary("M2")]))
'''
        ),
        _markdown("## 7. M3/M4 prompt and demonstration metadata"),
        _code(
            r'''llm_rows = []
for method in ("M3", "M4"):
    details = llm_config.get("methods", {}).get(method, {})
    llm_rows.append({
        "method": method,
        "experiment_id": details.get("experiment_id", "PENDING"),
        "prompt_version": details.get("prompt_version", "PENDING"),
        "prompt_sha256": details.get("prompt_sha256", "PENDING"),
        "demonstration_count": details.get("demonstration_count", "PENDING"),
        "model_requested": llm_config.get("api_request", {}).get("model", "PENDING"),
    })
display(pd.DataFrame(llm_rows))

a1_bank = llm_config.get("methods", {}).get("M4", {}).get("evaluation_banks", {}).get("A1", {})
display(pd.DataFrame([{
    "M4_A1_ordered_search_ranks": a1_bank.get("ordered_search_ranks", "PENDING"),
    "membership_sha256": a1_bank.get("membership_sha256", "PENDING"),
}]))
'''
        ),
        _markdown("## 8. Canonical M1–M4 A1 comparison"),
        _code(
            r'''a1_primary = load_csv(
    METRICS_ROOT / "a1/amp_primary_results.csv",
    ("method", "macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard", "test_n"),
)
if not a1_primary.empty:
    order = {method: index for index, method in enumerate(EXPECTED_METHODS)}
    a1_primary = a1_primary.assign(_order=a1_primary["method"].map(order)).sort_values("_order").drop(columns="_order")
show_or_pending(a1_primary, "run the canonical evaluator after prediction artifacts are complete.")
'''
        ),
        _markdown("## 9. Visual summaries of canonical aggregate metrics"),
        _code(
            r'''if not a1_primary.empty:
    columns = ["macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard"]
    axes = a1_primary.set_index("method")[columns].plot.bar(
        subplots=True, layout=(2, 2), figsize=(12, 8), legend=False, ylim=(0, 1),
        title=["Macro-F1", "Micro-F1", "Exact-set accuracy", "Example Jaccard"],
    )
    plt.suptitle("A1 canonical evaluator metrics (descriptive only)")
    plt.tight_layout()
else:
    display(Markdown("> **Pending:** no canonical A1 aggregate table to plot."))
'''
        ),
        _markdown("## 10. Canonical per-label precision, recall, and F1"),
        _code(
            r'''a1_per_label = load_csv(
    METRICS_ROOT / "a1/amp_per_label.csv",
    ("method", "label_id", "family", "support", "precision", "recall", "f1", "status"),
)
show_or_pending(a1_per_label)
'''
        ),
        _markdown("## 11. Canonical bootstrap confidence intervals"),
        _code(
            r'''a1_bootstrap = load_csv(
    METRICS_ROOT / "a1/amp_bootstrap_cis.csv",
    ("method", "metric", "estimate", "ci_lower", "ci_upper", "n_resamples", "seed"),
)
show_or_pending(a1_bootstrap)
'''
        ),
        _markdown("## 12. Fixed 0.50-threshold sensitivity (M1/M2 only)"),
        _code(
            r'''threshold_sensitivity = load_csv(
    METRICS_ROOT / "amp_threshold_0_50_sensitivity.csv",
    ("method", "evaluation", "prediction_variant", "macro_f1", "micro_f1"),
)
a1_sensitivity = threshold_sensitivity.loc[threshold_sensitivity.get("evaluation", pd.Series(dtype=str)).eq("A1")] if not threshold_sensitivity.empty else threshold_sensitivity
show_or_pending(a1_sensitivity)
'''
        ),
        _markdown(
            """## 13. Technical observations for researcher review

Add descriptive, non-speculative notes here after the canonical completion gate passes. Do not use test-set observations to alter prompts, demonstrations, thresholds, preprocessing, architecture, or the frozen metric protocol. Paper-level Results/Discussion conclusions are intentionally not generated by this notebook."""
        ),
    ]
    return _notebook(name, cells)


def _a2_notebook() -> dict[str, Any]:
    name = PRIMARY_NOTEBOOK_NAMES[1]
    cells = [
        _markdown(
            """# A2 AMP Results — Jurisdiction-Held-Out Test Sets

This notebook displays canonical outputs from `src/experiments/11_evaluate_amp.py`. It does not recompute metrics or bootstrap intervals. A2 uses SHERLOC Legacy Keywords as a **silver reference**. Where the pooled A2 reference support for `PURPOSE_REMOVAL_OF_ORGANS` is zero, its per-label F1 is **N/A** and macro-F1 is calculated over the 16 supported labels by the evaluator; all 17 outputs still enter micro/set metrics.

No scientific conclusions or protocol changes are generated here."""
        ),
        _markdown("## 1. Setup and canonical-artifact contract"),
        _code(COMMON_SETUP),
        _markdown("## 2. Artifact availability and completion gate"),
        _code(AVAILABILITY_CELL),
        _markdown("## 3. A2 fold composition and held-out jurisdictions"),
        _code(
            r'''a2_split_path = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
a2_split = load_csv(a2_split_path, ("search_rank", "fold_id", "role", "jurisdiction", "heldout_jurisdiction"))
if not a2_split.empty:
    fold_composition = (
        a2_split.groupby(["fold_id", "role"], dropna=False)
        .agg(cases=("search_rank", "size"), jurisdictions=("jurisdiction", "nunique"))
        .reset_index()
    )
    display(fold_composition)
    heldout = (
        a2_split.loc[a2_split["role"].eq("TEST"), ["fold_id", "jurisdiction"]]
        .drop_duplicates()
        .sort_values(["fold_id", "jurisdiction"])
    )
    display(heldout)
    display(pd.DataFrame([{"split_sha256": sha256_file(a2_split_path)}]))
else:
    show_or_pending(a2_split)
'''
        ),
        _markdown("## 4. Pooled OOD label support and Organ Removal status"),
        _code(
            r'''a2_per_label = load_csv(
    METRICS_ROOT / "a2/amp_per_label.csv",
    ("method", "label_id", "family", "support", "precision", "recall", "f1", "status", "included_in_macro_f1"),
)
if not a2_per_label.empty:
    support_view = a2_per_label[["method", "label_id", "family", "support", "status", "included_in_macro_f1"]]
    display(support_view)
    organ_rows = a2_per_label.loc[a2_per_label["label_id"].eq("PURPOSE_REMOVAL_OF_ORGANS")]
    display(Markdown("**Organ Removal evaluator rows:**"))
    display(organ_rows)
    display(pd.DataFrame([{
        "manifest_rule": evaluation_manifest.get("evaluations", {}).get("A2", {}).get("organ_removal_rule", "PENDING"),
        "macro_label_count": evaluation_manifest.get("evaluations", {}).get("A2", {}).get("macro_label_count", "PENDING"),
    }]))
else:
    show_or_pending(a2_per_label)
'''
        ),
        _markdown("## 5. Canonical per-fold results"),
        _code(
            r'''a2_per_fold = load_csv(
    METRICS_ROOT / "a2/amp_per_fold.csv",
    ("method", "fold", "macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard", "test_n"),
)
show_or_pending(a2_per_fold)
'''
        ),
        _markdown("## 6. Canonical pooled OOD results"),
        _code(
            r'''a2_primary = load_csv(
    METRICS_ROOT / "a2/amp_primary_results.csv",
    ("method", "fold_1_macro_f1", "fold_2_macro_f1", "fold_3_macro_f1", "pooled_ood_macro_f1", "pooled_micro_f1", "pooled_exact_set_accuracy", "pooled_example_jaccard", "test_n"),
)
if not a2_primary.empty:
    order = {method: index for index, method in enumerate(EXPECTED_METHODS)}
    a2_primary = a2_primary.assign(_order=a2_primary["method"].map(order)).sort_values("_order").drop(columns="_order")
show_or_pending(a2_primary)
'''
        ),
        _markdown("## 7. Canonical per-jurisdiction results"),
        _code(
            r'''a2_per_jurisdiction = load_csv(
    METRICS_ROOT / "a2/amp_per_jurisdiction.csv",
    ("method", "jurisdiction", "fold", "macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard", "test_n"),
)
show_or_pending(a2_per_jurisdiction)
'''
        ),
        _markdown("## 8. Canonical A1 → A2 aggregate deltas"),
        _code(
            r'''a1_a2_deltas = load_csv(
    METRICS_ROOT / "amp_a1_to_a2_deltas.csv",
    ("method", "delta_macro_f1_a2_minus_a1", "delta_micro_f1_a2_minus_a1", "delta_exact_set_a2_minus_a1", "delta_example_jaccard_a2_minus_a1", "significance_claim"),
)
show_or_pending(a1_a2_deltas)
'''
        ),
        _markdown("## 9. Per-label IID/OOD comparison"),
        _code(
            r'''a1_per_label = load_csv(
    METRICS_ROOT / "a1/amp_per_label.csv",
    ("method", "label_id", "f1", "support", "status"),
)
if not a1_per_label.empty and not a2_per_label.empty:
    # This is a side-by-side view of already-computed evaluator values. It does
    # not reconstruct F1 or assert statistical significance.
    per_label_comparison = a1_per_label[["method", "label_id", "f1", "support", "status"]].merge(
        a2_per_label[["method", "label_id", "f1", "support", "status"]],
        on=["method", "label_id"], how="outer", suffixes=("_a1", "_a2"), validate="one_to_one",
    )
    display(per_label_comparison)
else:
    display(Markdown("> **Pending:** both canonical A1 and A2 per-label tables are required."))
'''
        ),
        _markdown("## 10. Canonical pooled bootstrap confidence intervals"),
        _code(
            r'''a2_bootstrap = load_csv(
    METRICS_ROOT / "a2/amp_bootstrap_cis.csv",
    ("method", "metric", "estimate", "ci_lower", "ci_upper", "n_resamples", "seed"),
)
show_or_pending(a2_bootstrap)
'''
        ),
        _markdown("## 11. Descriptive visual summaries"),
        _code(
            r'''if not a2_primary.empty:
    pooled_columns = ["pooled_ood_macro_f1", "pooled_micro_f1", "pooled_exact_set_accuracy", "pooled_example_jaccard"]
    a2_primary.set_index("method")[pooled_columns].plot.bar(
        subplots=True, layout=(2, 2), figsize=(12, 8), legend=False, ylim=(0, 1),
        title=["Pooled macro-F1", "Pooled micro-F1", "Pooled exact-set", "Pooled Jaccard"],
    )
    plt.suptitle("A2 canonical evaluator metrics (descriptive only)")
    plt.tight_layout()
else:
    display(Markdown("> **Pending:** no canonical A2 aggregate table to plot."))
'''
        ),
        _markdown(
            """## 12. Technical observations for researcher review

Record short technical observations only after the completion gate passes. A negative A1→A2 delta is not automatically statistically significant; the canonical delta table explicitly records `NOT_TESTED_DO_NOT_INFER`. Do not change later folds, prompts, demonstrations, models, or thresholds based on these test outcomes."""
        ),
    ]
    return _notebook(name, cells)


def _error_notebook() -> dict[str, Any]:
    name = PRIMARY_NOTEBOOK_NAMES[2]
    cells = [
        _markdown(
            """# AMP Error Analysis — Manual Inspection Workspace

This exploratory notebook loads canonical case-level error rows produced by `src/experiments/11_evaluate_amp.py`. It supports transparent filtering and disagreement inspection without recalculating benchmark metrics.

Selections here are post-hoc and must not be used to alter or rerun the frozen primary benchmark. Case examples chosen for publication require a documented, non-cherry-picked sampling rule."""
        ),
        _markdown("## 1. Setup and canonical-artifact contract"),
        _code(COMMON_SETUP),
        _markdown("## 2. Artifact availability and completion gate"),
        _code(AVAILABILITY_CELL),
        _markdown("## 3. Load canonical case-level evaluator outputs"),
        _code(
            r'''error_columns = (
    "method", "case_id", "search_rank", "jurisdiction", "split", "fold", "fact_summary",
    "silver_reference_amp_json", "predicted_amp_json", "false_positive_labels_json",
    "false_negative_labels_json", "exact_set_correct", "example_jaccard", "truncated_input",
)
a1_errors = load_csv(METRICS_ROOT / "a1/amp_case_level_errors.csv", error_columns).assign(evaluation="A1")
a2_errors = load_csv(METRICS_ROOT / "a2/amp_case_level_errors.csv", error_columns).assign(evaluation="A2")
case_errors = pd.concat([a1_errors, a2_errors], ignore_index=True)
case_errors["narrative_char_count"] = case_errors["fact_summary"].fillna("").str.len()
if not case_errors.empty:
    display(case_errors.groupby(["evaluation", "method"], dropna=False).size().rename("rows").reset_index())
else:
    show_or_pending(case_errors)
'''
        ),
        _markdown("## 4. Reusable manual filters"),
        _code(
            r'''# Edit these controls, then rerun this cell. None of them changes a model or metric.
EVALUATION = "A1"          # "A1", "A2", or None
METHOD = None              # "M1", "M2", "M3", "M4", or None
FOLD = None                # 1, 2, 3, or None
JURISDICTION = None        # exact string or None
EXACT_SET_FAILURES_ONLY = False
TRUNCATED_M2_ONLY = False
MIN_JACCARD = None
MAX_JACCARD = None
LABEL_ID = None            # exact ontology ID found in FP or FN arrays, or None
SORT = "example_jaccard"   # example_jaccard, narrative_char_count, or search_rank
ASCENDING = True
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
if TRUNCATED_M2_ONLY:
    filtered = filtered.loc[filtered["method"].eq("M2") & filtered["truncated_input"].eq(1)]
if MIN_JACCARD is not None:
    filtered = filtered.loc[filtered["example_jaccard"].ge(MIN_JACCARD)]
if MAX_JACCARD is not None:
    filtered = filtered.loc[filtered["example_jaccard"].le(MAX_JACCARD)]
if LABEL_ID is not None:
    def contains_label(value):
        return LABEL_ID in json.loads(value)
    filtered = filtered.loc[
        filtered["false_positive_labels_json"].map(contains_label)
        | filtered["false_negative_labels_json"].map(contains_label)
    ]

inspection_columns = [
    "evaluation", "method", "search_rank", "jurisdiction", "fold", "narrative_char_count",
    "exact_set_correct", "example_jaccard", "truncated_input", "silver_reference_amp_json",
    "predicted_amp_json", "false_positive_labels_json", "false_negative_labels_json", "fact_summary",
]
display(filtered.sort_values(SORT, ascending=ASCENDING)[inspection_columns].head(MAX_ROWS))
'''
        ),
        _markdown("## 5. Prediction disagreements among M1–M4"),
        _code(
            r'''if not case_errors.empty:
    disagreement_key = ["evaluation", "search_rank", "jurisdiction", "fold"]
    prediction_matrix = case_errors.pivot_table(
        index=disagreement_key,
        columns="method",
        values="predicted_amp_json",
        aggfunc="first",
    ).reset_index()
    method_columns = [method for method in EXPECTED_METHODS if method in prediction_matrix.columns]
    prediction_matrix["distinct_prediction_count"] = prediction_matrix[method_columns].nunique(axis=1, dropna=True)
    disagreements = prediction_matrix.loc[prediction_matrix["distinct_prediction_count"].gt(1)]
    display(disagreements.sort_values(["evaluation", "search_rank"]).head(100))
else:
    display(Markdown("> **Pending:** canonical case-level rows are unavailable."))
'''
        ),
        _markdown("## 6. M3 versus M4 and supervised versus LLM views"),
        _code(
            r'''if "prediction_matrix" in globals() and not prediction_matrix.empty:
    if {"M3", "M4"}.issubset(prediction_matrix.columns):
        m3_m4 = prediction_matrix.loc[
            prediction_matrix["M3"].notna()
            & prediction_matrix["M4"].notna()
            & prediction_matrix["M3"].ne(prediction_matrix["M4"])
        ]
        display(Markdown("**M3/M4 prediction disagreements:**"))
        display(m3_m4.head(100))
    else:
        display(Markdown("> **Pending:** both M3 and M4 canonical rows are required."))

    if set(EXPECTED_METHODS).issubset(prediction_matrix.columns):
        supervised_llm = prediction_matrix.loc[
            prediction_matrix[["M1", "M2"]].nunique(axis=1).eq(1)
            & prediction_matrix[["M3", "M4"]].nunique(axis=1).eq(1)
            & prediction_matrix["M1"].ne(prediction_matrix["M3"])
        ]
        display(Markdown("**Cases where supervised methods agree, LLM methods agree, and the groups disagree:**"))
        display(supervised_llm.head(100))
    else:
        display(Markdown("> **Pending:** all four canonical method rows are required for the grouped view."))
'''
        ),
        _markdown("## 7. Canonical jurisdiction performance and M2 truncation cases"),
        _code(
            r'''a2_jurisdiction = load_csv(
    METRICS_ROOT / "a2/amp_per_jurisdiction.csv",
    ("method", "jurisdiction", "fold", "macro_f1", "micro_f1", "exact_set_accuracy", "example_jaccard", "test_n"),
)
show_or_pending(a2_jurisdiction)

if not case_errors.empty:
    m2_truncated = case_errors.loc[case_errors["method"].eq("M2") & case_errors["truncated_input"].eq(1)]
    display(Markdown("**Canonical M2 rows marked as truncated:**"))
    display(m2_truncated[inspection_columns].sort_values(["evaluation", "search_rank"]))
'''
        ),
        _markdown("## 8. Rare-label inspection"),
        _code(
            r'''a1_label = load_csv(METRICS_ROOT / "a1/amp_per_label.csv", ("method", "label_id", "support", "status"))
a2_label = load_csv(METRICS_ROOT / "a2/amp_per_label.csv", ("method", "label_id", "support", "status"))
label_support = pd.concat([a1_label.assign(evaluation="A1"), a2_label.assign(evaluation="A2")], ignore_index=True)
if not label_support.empty:
    display(label_support.sort_values(["evaluation", "support", "label_id"]))
display(Markdown(
    "Set `LABEL_ID` in the manual-filter cell to inspect errors involving a rare label. "
    "In A2, zero-support Organ Removal is N/A for per-label F1 but false positives remain visible in case-level errors."
))
'''
        ),
        _markdown(
            """## 9. Researcher notes and post-hoc labeling

Record the filter settings and selection rule for every saved case set. Mark any analysis developed after viewing test outcomes as **post-hoc exploratory analysis**. Do not use this notebook to tune or rerun the frozen benchmark, and do not infer narrative-grounded correctness from disagreement with the silver reference."""
        ),
    ]
    return _notebook(name, cells)


def _auxiliary_notebook() -> dict[str, Any]:
    name = AUXILIARY_NOTEBOOK_NAME
    cells = [
        _markdown(
            """# Auxiliary Feature Results — Exploratory Only

This optional notebook is separate from the 17-output primary AMP benchmark. It loads only machine-readable artifacts under `outputs/metrics/auxiliary/` and must never add Geographic Form, Victim Multiplicity, Sector, or Child/Minor targets to the core M3/M4 prompt.

All Form/Sector results are **exploratory silver-reference results**. Multiplicity results remain provisional unless backed by completed human confirmation. This notebook does not calculate metrics or generate scientific conclusions."""
        ),
        _markdown("## 1. Setup"),
        _code(COMMON_SETUP),
        _markdown("## 2. Auxiliary artifact inventory"),
        _code(
            r'''auxiliary_root = METRICS_ROOT / "auxiliary"
if auxiliary_root.is_dir():
    auxiliary_files = sorted(path for path in auxiliary_root.rglob("*") if path.is_file())
    display(pd.DataFrame({"artifact": [str(path.relative_to(REPO_ROOT)) for path in auxiliary_files]}))
else:
    display(Markdown(
        "> **Pending by protocol:** `outputs/metrics/auxiliary/` does not exist. "
        "Primary M1-M4 A1/A2 AMP completion is required before auxiliary execution."
    ))
'''
        ),
        _markdown("## 3. Protocol status"),
        _code(
            r'''auxiliary_status = pd.DataFrame([
    {"feature": "Geographic Form", "required_label": "EXPLORATORY SILVER-REFERENCE FORM RESULTS", "status": "PENDING_OR_LOAD_FROM_CANONICAL_ARTIFACT"},
    {"feature": "Victim Multiplicity", "required_label": "PROVISIONAL unless human-confirmed", "status": "PENDING_HUMAN_CONFIRMATION"},
    {"feature": "Sector", "required_label": "EXPLORATORY SILVER-REFERENCE SECTOR RESULTS", "status": "OPTIONAL_AFTER_PRIMARY"},
    {"feature": "Child/minor", "required_label": "EXPLORATORY", "status": "DEFERRED"},
])
display(auxiliary_status)
'''
        ),
        _markdown(
            """## 4. Researcher review

Add loaders for specific canonical auxiliary evaluator tables only after those pipelines exist. Do not derive an independent split: intersect auxiliary eligibility with the frozen A1/A2 assignments."""
        ),
    ]
    return _notebook(name, cells)


def build_notebooks(*, include_auxiliary: bool = False) -> dict[str, dict[str, Any]]:
    """Return deterministic notebook documents keyed by output filename."""
    notebooks = {
        PRIMARY_NOTEBOOK_NAMES[0]: _a1_notebook(),
        PRIMARY_NOTEBOOK_NAMES[1]: _a2_notebook(),
        PRIMARY_NOTEBOOK_NAMES[2]: _error_notebook(),
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
        help="Also generate the post-primary auxiliary reporting notebook.",
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
