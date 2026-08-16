"""Synthetic regression tests for canonical single-reviewer Evaluation B analysis."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sherloc_test_mpl"))
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "src/experiments/18_evaluate_evaluation_b.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluation_b_analysis", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Evaluation B analysis module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS = load_module()


def human_case(
    case_id: str,
    rank: int,
    *,
    status: str = "SUBSTANTIVE",
    acts=(),
    means=(),
    purposes=(),
):
    fact_summary = f"Narrative {rank}."
    return ANALYSIS.HumanCase(
        reliability_case_id=case_id,
        search_rank=rank,
        canonical_url=f"https://example.test/case/{rank}",
        input_sha256=ANALYSIS._sha256_text(fact_summary),
        jurisdiction=f"Jurisdiction {rank}",
        fact_summary=fact_summary,
        review_status=status,
        substantive_amp_evaluable=status == "SUBSTANTIVE",
        auxiliary_evaluable=status == "SUBSTANTIVE",
        annotation_notes="Abstain" if status == "ABSTAIN" else "",
        labels={"ACT": tuple(acts), "MEANS": tuple(means), "PURPOSE": tuple(purposes)},
        geographic_form=("Internal",) if status == "SUBSTANTIVE" else (),
        organized_criminal_group=False,
        organized_criminal_group_evaluable=status == "SUBSTANTIVE",
        multiplicity="SINGLE" if status == "SUBSTANTIVE" else "Not Applicable",
        child="FALSE" if status == "SUBSTANTIVE" else "Not Applicable",
    )


def silver_case(
    human,
    *,
    acts=None,
    means=None,
    purposes=None,
    family_available=None,
    primary=True,
):
    return ANALYSIS.SilverCase(
        reliability_case_id=human.reliability_case_id,
        search_rank=human.search_rank,
        labels={
            "ACT": tuple(human.labels["ACT"] if acts is None else acts),
            "MEANS": tuple(human.labels["MEANS"] if means is None else means),
            "PURPOSE": tuple(human.labels["PURPOSE"] if purposes is None else purposes),
        },
        family_available=(
            {family: True for family in ANALYSIS.FAMILIES}
            if family_available is None
            else dict(family_available)
        ),
        geographic_form=("Internal",),
        organized_criminal_group=False,
        multiplicity="SINGLE",
        child="FALSE",
        primary_amp_cohort_member=primary,
    )


def prediction(method: str, human, labels=()):
    return ANALYSIS.PredictionCase(
        method=method,
        reliability_case_id=human.reliability_case_id,
        search_rank=human.search_rank,
        status="SUCCESS_VALIDATED",
        labels=tuple(label for label in ANALYSIS.AMP_LABEL_IDS if label in set(labels)),
    )


def synthetic_bundle():
    cases = [
        human_case(
            "HRV1-001",
            1,
            acts=("ACT_RECRUITMENT",),
            means=("MEANS_DECEPTION",),
            purposes=("PURPOSE_FORCED_LABOUR_OR_SERVICES",),
        ),
        human_case(
            "HRV1-002",
            2,
            acts=(),
            means=("MEANS_FRAUD",),
            purposes=(),
        ),
        human_case(
            "HRV1-003",
            3,
            acts=("ACT_TRANSPORTATION",),
            means=(),
            purposes=("PURPOSE_SEXUAL_EXPLOITATION",),
        ),
        human_case(
            "HRV1-004",
            4,
            acts=("ACT_TRANSFER",),
            means=("MEANS_ABDUCTION",),
            purposes=("PURPOSE_OTHER",),
        ),
        human_case("HRV1-005", 5, status="ABSTAIN"),
        human_case("HRV1-006", 6, status="ABSTAIN"),
    ]
    human = {item.reliability_case_id: item for item in cases}
    silver = {
        "HRV1-001": silver_case(
            human["HRV1-001"], acts=("ACT_RECRUITMENT", "ACT_TRANSPORTATION")
        ),
        "HRV1-002": silver_case(
            human["HRV1-002"], acts=("ACT_RECEIPT",), purposes=()
        ),
        "HRV1-003": silver_case(
            human["HRV1-003"], acts=(), means=("MEANS_DECEPTION",)
        ),
        "HRV1-004": silver_case(human["HRV1-004"]),
        "HRV1-005": silver_case(human["HRV1-005"], acts=(), means=(), purposes=()),
        "HRV1-006": silver_case(human["HRV1-006"], acts=(), means=(), purposes=()),
    }
    predictions = {method: {} for method in ANALYSIS.METHODS}
    for method in ANALYSIS.METHODS:
        for case in cases:
            labels = [label for family in ANALYSIS.FAMILIES for label in case.labels[family]]
            if method == "M1" and case.reliability_case_id == "HRV1-002":
                labels = ["ACT_RECRUITMENT", "MEANS_FRAUD"]
            if method == "M2" and case.review_status == "ABSTAIN":
                labels = ["ACT_RECEIPT"]
            if method == "M3" and case.reliability_case_id == "HRV1-003":
                labels = ["ACT_TRANSPORTATION"]
            if method == "M4" and case.reliability_case_id == "HRV1-004":
                continue  # frozen demonstration overlap: intentionally unevaluated
            predictions[method][case.reliability_case_id] = prediction(method, case, labels)
    return human, silver, predictions, {"HRV1-004"}


class EvaluationBAnalysisTest(unittest.TestCase):
    def test_actual_frozen_reference_and_silver_contract_load(self) -> None:
        human = ANALYSIS.load_human_reference(
            REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
        )
        self.assertEqual(len(human), 61)
        self.assertEqual(
            sum(case.review_status == "SUBSTANTIVE" for case in human.values()), 55
        )
        self.assertEqual(sum(case.review_status == "ABSTAIN" for case in human.values()), 6)
        self.assertTrue(any(case.labels["PURPOSE"] for case in human.values()))
        silver = ANALYSIS.load_silver_reference(
            REPO_ROOT / "data/annotations/reliability_sample_100_reference_key.csv",
            REPO_ROOT / "data/annotations/reliability_sample_100.csv",
            REPO_ROOT / "data/processed/sherloc_benchmark_v1.csv",
            human,
        )
        self.assertEqual(set(silver), set(human))
        substantive_ids = [
            case_id
            for case_id, case in human.items()
            if case.review_status == "SUBSTANTIVE"
        ]
        self.assertEqual(
            {
                family: sum(silver[case_id].family_available[family] for case_id in substantive_ids)
                for family in ANALYSIS.FAMILIES
            },
            {"ACT": 54, "MEANS": 54, "PURPOSE": 55},
        )
        self.assertEqual(
            sum(all(silver[case_id].family_available.values()) for case_id in substantive_ids),
            54,
        )
        self.assertEqual(
            silver["HRV1-050"].family_available,
            {"ACT": False, "MEANS": False, "PURPOSE": True},
        )
        self.assertFalse(
            any(
                case.multiplicity == "NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT"
                or case.child == "NOT_APPLICABLE_OUTSIDE_PRIMARY_COHORT"
                for case in silver.values()
            )
        )
        source_manifest = json.loads(
            (
                REPO_ROOT
                / "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
            ).read_text(encoding="utf-8")
        )
        qc_summary = json.loads(
            (
                REPO_ROOT
                / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
            ).read_text(encoding="utf-8")
        )
        membership_manifest_path = (
            REPO_ROOT
            / "outputs/analysis/evaluation_b/eval_b_membership_manifest.json"
        )
        membership_manifest = json.loads(
            membership_manifest_path.read_text(encoding="utf-8")
        )
        source = ANALYSIS.validate_human_reference_provenance(
            source_manifest,
            qc_summary,
            human,
            human_reference_path=(
                REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
            ),
            source_manifest_path=(
                REPO_ROOT
                / "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
            ),
            qc_summary_path=(
                REPO_ROOT
                / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
            ),
            membership_manifest=membership_manifest,
        )
        self.assertEqual(
            ANALYSIS._sha256_file(source), source_manifest["source"]["sha256"]
        )
        self.assertEqual(
            ANALYSIS.retained_membership_sha256(human),
            membership_manifest["retained_membership_sha256"],
        )
        stale_manifest = json.loads(json.dumps(membership_manifest))
        stale_manifest["human_reference"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ANALYSIS.EvaluationBAnalysisError, "human reference SHA-256"
        ):
            ANALYSIS.validate_human_reference_provenance(
                source_manifest,
                qc_summary,
                human,
                human_reference_path=(
                    REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
                ),
                source_manifest_path=(
                    REPO_ROOT
                    / "outputs/analysis/evaluation_b/human_annotation_source_manifest.json"
                ),
                qc_summary_path=(
                    REPO_ROOT
                    / "outputs/analysis/evaluation_b/human_annotation_qc_summary.json"
                ),
                membership_manifest=stale_manifest,
            )

    def test_leakage_audit_requires_all_four_exclusion_gates(self) -> None:
        human = ANALYSIS.load_human_reference(
            REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
        )
        membership_sha256 = ANALYSIS.retained_membership_sha256(human)
        audit_path = (
            REPO_ROOT
            / "outputs/analysis/evaluation_b/eval_b_training_exclusion_audit.csv"
        )
        ANALYSIS.validate_leakage_audit(
            audit_path,
            set(human),
            expected_membership_sha256=membership_sha256,
        )
        rows = ANALYSIS._read_csv(audit_path)
        rows[0]["removed_from_eval_b_threshold_tuning"] = "FALSE"
        with tempfile.TemporaryDirectory() as directory:
            stale_path = Path(directory) / "stale_audit.csv"
            with stale_path.open("w", encoding="utf-8", newline="") as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(
                ANALYSIS.EvaluationBAnalysisError,
                "removed_from_eval_b_threshold_tuning",
            ):
                ANALYSIS.validate_leakage_audit(
                    stale_path,
                    set(human),
                    expected_membership_sha256=membership_sha256,
                )

    def test_common_membership_is_exact_and_excludes_demo_overlap(self) -> None:
        human, _silver, predictions, overlap = synthetic_bundle()
        common, abstain = ANALYSIS.build_common_membership(human, predictions, overlap)
        self.assertEqual(common, ["HRV1-001", "HRV1-002", "HRV1-003"])
        self.assertEqual(abstain, ["HRV1-005", "HRV1-006"])
        predictions["M3"].pop("HRV1-002")
        with self.assertRaisesRegex(ANALYSIS.EvaluationBAnalysisError, "missing common"):
            ANALYSIS.build_common_membership(human, predictions, overlap)

    def test_empty_reference_and_nonempty_cpmr_are_distinct(self) -> None:
        human, _silver, predictions, _overlap = synthetic_bundle()
        ids = ["HRV1-001", "HRV1-002", "HRV1-003"]
        reference = ANALYSIS._matrix_from_cases(ids, human, kind="human")
        predicted = ANALYSIS._matrix_from_cases(ids, predictions["M1"], kind="M1")
        act = ANALYSIS._family_diagnostics(reference, predicted, "ACT")
        self.assertEqual(act["nonempty_reference_n"], 2)
        self.assertEqual(act["empty_reference_n"], 1)
        self.assertEqual(act["empty_reference_correct_empty_count"], 0)
        self.assertEqual(act["empty_reference_correct_empty_rate"], 0.0)
        self.assertEqual(act["cpmr"], 2 / 3)
        self.assertEqual(act["cpmr_nonempty_reference"], 1.0)

    def test_silver_human_categories_and_neutral_counts(self) -> None:
        human, silver, _predictions, _overlap = synthetic_bundle()
        ids = ["HRV1-001", "HRV1-002", "HRV1-003"]
        summary, per_label, case_rows = ANALYSIS.compare_silver_human(ids, human, silver)
        act_rows = {row["reliability_case_id"]: row for row in case_rows if row["family"] == "ACT"}
        self.assertEqual(act_rows["HRV1-001"]["category"], "SILVER_BROADER")
        self.assertEqual(act_rows["HRV1-002"]["category"], "SILVER_BROADER")
        self.assertEqual(act_rows["HRV1-003"]["category"], "HUMAN_BROADER")
        act_summary = next(row for row in summary if row["family"] == "ACT")
        self.assertEqual(act_summary["shared_label_count"], 1)
        self.assertEqual(act_summary["silver_only_label_count"], 2)
        self.assertEqual(act_summary["human_only_label_count"], 1)
        self.assertEqual(act_summary["silver_only_rate_of_silver_labels"], 2 / 3)
        self.assertEqual(act_summary["human_only_rate_of_human_labels"], 1 / 2)
        self.assertEqual(len(per_label), 17)

    def test_partial_silver_availability_never_becomes_an_empty_reference(self) -> None:
        human, silver, predictions, overlap = synthetic_bundle()
        partial = human["HRV1-002"]
        silver["HRV1-002"] = silver_case(
            partial,
            acts=(),
            family_available={"ACT": False, "MEANS": True, "PURPOSE": True},
            primary=False,
        )
        tables, metadata = ANALYSIS.build_analysis(
            human,
            silver,
            predictions,
            overlap,
            bootstrap_resamples=10,
            bootstrap_seed=3,
        )
        self.assertEqual(metadata["common_substantive_n"], 3)
        self.assertEqual(metadata["silver_human_substantive_n"], 4)
        self.assertEqual(metadata["dual_reference_complete_amp_n"], 2)
        self.assertTrue(
            all(row["n"] == 3 for row in tables["eval_b_main_results.csv"])
        )
        m1_main = next(
            row for row in tables["eval_b_main_results.csv"] if row["method"] == "M1"
        )
        self.assertEqual(m1_main["exact_set"], 2 / 3)
        silver_summary = {
            row["family"]: row for row in tables["silver_vs_human_summary.csv"]
        }
        self.assertEqual(silver_summary["ACT"]["substantive_n"], 4)
        self.assertEqual(silver_summary["ACT"]["comparable_n"], 3)
        self.assertEqual(silver_summary["ACT"]["silver_reference_unavailable_n"], 1)
        self.assertEqual(silver_summary["MEANS"]["comparable_n"], 4)
        self.assertEqual(silver_summary["PURPOSE"]["comparable_n"], 4)
        unavailable = next(
            row
            for row in tables["silver_vs_human_case_level.csv"]
            if row["reliability_case_id"] == "HRV1-002" and row["family"] == "ACT"
        )
        self.assertEqual(unavailable["silver_reference_available"], 0)
        self.assertEqual(unavailable["category"], "SILVER_REFERENCE_UNAVAILABLE")
        self.assertIsNone(unavailable["exact_set_concordance"])
        self.assertTrue(
            all(
                row["n"] == 2
                and row["dual_reference_n"] == 2
                and row["excluded_incomplete_silver_reference_n"] == 1
                for row in tables["model_silver_vs_human_metric_comparison.csv"]
            )
        )
        m1_dual_exact = next(
            row
            for row in tables["model_silver_vs_human_metric_comparison.csv"]
            if row["method"] == "M1"
            and row["metric_scope"] == "OVERALL"
            and row["metric"] == "exact_set"
        )
        self.assertEqual(m1_dual_exact["human_grounded_value"], 1.0)
        breadth = tables["eval_b_prediction_breadth.csv"]
        self.assertTrue(
            all(
                row["n"] == 3
                and row["silver_act_reference_available_n"] == 2
                and row["silver_means_reference_available_n"] == 3
                and row["silver_purpose_reference_available_n"] == 3
                and row["complete_silver_amp_reference_n"] == 2
                for row in breadth
            )
        )
        case_row = next(
            row
            for row in tables["human_grounded_case_level_errors.csv"]
            if row["reliability_case_id"] == "HRV1-002"
        )
        self.assertEqual(case_row["silver_act_reference_available"], 0)
        self.assertEqual(
            case_row["silver_human_act_category"], "SILVER_REFERENCE_UNAVAILABLE"
        )
        self.assertEqual(case_row["complete_silver_amp_reference_available"], 0)

    def test_abstain_is_a_separate_diagnostic_not_negative_accuracy(self) -> None:
        human, _silver, predictions, _overlap = synthetic_bundle()
        summary, cases = ANALYSIS.evaluate_abstain(
            ["HRV1-005", "HRV1-006"], human, predictions
        )
        by_method = {row["method"]: row for row in summary}
        self.assertEqual(by_method["M1"]["all_amp_empty_rate"], 1.0)
        self.assertEqual(by_method["M2"]["all_amp_empty_rate"], 0.0)
        self.assertEqual(
            by_method["M2"][
                "total_unsupported_predicted_label_count_under_abstention_interpretation"
            ],
            2,
        )
        self.assertTrue(
            all("NOT_ALL_NEGATIVE_ACCURACY" in row["interpretation"] for row in cases)
        )

    def test_bootstrap_is_deterministic_and_includes_family_cpmr(self) -> None:
        human, _silver, predictions, _overlap = synthetic_bundle()
        ids = ["HRV1-001", "HRV1-002", "HRV1-003"]
        reference = ANALYSIS._matrix_from_cases(ids, human, kind="human")
        predicted = ANALYSIS._matrix_from_cases(ids, predictions["M1"], kind="M1")
        first = ANALYSIS.bootstrap_intervals(reference, predicted, n_resamples=25, seed=7)
        second = ANALYSIS.bootstrap_intervals(reference, predicted, n_resamples=25, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            {row["bootstrap_method"] for row in first}, {"PERCENTILE_LINEAR"}
        )
        self.assertTrue(all("method" not in row for row in first))
        self.assertEqual(
            {row["metric"] for row in first},
            {
                "macro_f1",
                "micro_f1",
                "exact_set",
                "jaccard",
                "act_cpmr",
                "means_cpmr",
                "purpose_cpmr",
            },
        )

    def test_complete_bundle_tables_report_and_four_figures(self) -> None:
        human, silver, predictions, overlap = synthetic_bundle()
        tables, metadata = ANALYSIS.build_analysis(
            human,
            silver,
            predictions,
            overlap,
            bootstrap_resamples=20,
            bootstrap_seed=17,
        )
        self.assertEqual(set(tables), set(ANALYSIS.OUTPUT_TABLE_NAMES))
        self.assertEqual(metadata["common_substantive_n"], 3)
        self.assertEqual(len(tables["eval_b_main_results.csv"]), 4)
        self.assertEqual(len(tables["human_grounded_case_level_errors.csv"]), 3)
        bootstrap_rows = tables["eval_b_bootstrap_cis.csv"]
        self.assertEqual(len(bootstrap_rows), 28)
        self.assertEqual(
            {row["bootstrap_method"] for row in bootstrap_rows},
            {"PERCENTILE_LINEAR"},
        )
        self.assertEqual(
            {method: sum(row["method"] == method for row in bootstrap_rows) for method in ANALYSIS.METHODS},
            {method: 7 for method in ANALYSIS.METHODS},
        )
        comparison = tables["model_silver_vs_human_metric_comparison.csv"]
        self.assertTrue(
            all(
                "silver_reference_dense_rank" in row
                and "human_grounded_dense_rank" in row
                and "rank_changed" in row
                for row in comparison
            )
        )
        report = ANALYSIS.render_report(
            tables,
            metadata,
            {"row_count": 100},
            {"reviewed_n": 6, "unreviewed_n": 94, "skip_n": 0},
            {
                "M1": {"train_n": 100},
                "M2": {"train_n": 100},
                "M3": {"new_api_request_successes": 6, "reused_identical_requests": 0},
                "M4": {"new_api_request_successes": 5, "reused_identical_requests": 0},
            },
        )
        self.assertIn("single-reviewer human-grounded narrative reference", report)
        self.assertIn("reviewer-to-reviewer agreement could not be estimated", report)
        self.assertTrue(report.endswith(ANALYSIS.FINAL_REPORT_SENTENCE))
        self.assertIn("delta_human_minus_silver", report)
        for forbidden in ("two-reviewer gold", "adjudicated gold", "inter-annotator agreement"):
            self.assertNotIn(forbidden, report.lower())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            ANALYSIS.generate_figures(tables, target)
            observed = sorted(path.name for path in target.iterdir())
            self.assertEqual(observed, sorted(ANALYSIS.FIGURE_NAMES))
            self.assertTrue(all((target / name).stat().st_size > 1_000 for name in observed))
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as directory:
            target = Path(directory)
            output_dir = target / "analysis"
            figure_dir = target / "figures"
            report_path = target / "report.md"
            arguments = dict(
                output_dir=output_dir,
                figure_dir=figure_dir,
                report_path=report_path,
                report_text=report,
                input_hashes={"synthetic": "a" * 64},
            )
            ANALYSIS._write_outputs(tables, metadata, **arguments)
            first = {
                path.relative_to(target).as_posix(): path.read_bytes()
                for path in target.rglob("*")
                if path.is_file()
            }
            ANALYSIS._write_outputs(tables, metadata, **arguments)
            second = {
                path.relative_to(target).as_posix(): path.read_bytes()
                for path in target.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first, second)

    def test_prediction_loader_rejects_unvalidated_or_incomplete_membership(self) -> None:
        human, _silver, _predictions, _overlap = synthetic_bundle()
        subset = {key: human[key] for key in ("HRV1-001", "HRV1-002")}
        membership_sha256 = "a" * 64
        case = subset["HRV1-001"]
        base = {
            "method_id": "M1",
            "evaluation": "B",
            "reliability_case_id": "HRV1-001",
            "search_rank": 1,
            "canonical_url": case.canonical_url,
            "fact_summary": case.fact_summary,
            "input_sha256": case.input_sha256,
            "retained_membership_sha256": membership_sha256,
            "config_sha256": ANALYSIS.EXPECTED_SUPERVISED_CONFIG_SHA256["M1"],
            "human_labels_used_for_training_tuning_or_prediction": False,
            "predicted_labels": ["ACT_RECRUITMENT"],
        }
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "predictions.jsonl"
            target.write_text(
                json.dumps(
                    {
                        **base,
                        "status": "PARTIAL",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ANALYSIS.EvaluationBAnalysisError, "nonvalidated"):
                ANALYSIS.load_predictions(
                    "M1",
                    target,
                    subset,
                    set(subset),
                    expected_membership_sha256=membership_sha256,
                )

            target.write_text(
                json.dumps(
                    {
                        **base,
                        "status": "SUCCESS_VALIDATED",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ANALYSIS.EvaluationBAnalysisError, "membership mismatch"):
                ANALYSIS.load_predictions(
                    "M1",
                    target,
                    subset,
                    set(subset),
                    expected_membership_sha256=membership_sha256,
                )

            stale = {**base, "status": "SUCCESS_VALIDATED", "input_sha256": "b" * 64}
            target.write_text(json.dumps(stale) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ANALYSIS.EvaluationBAnalysisError, "input_sha256"):
                ANALYSIS.load_predictions(
                    "M1",
                    target,
                    {"HRV1-001": case},
                    {"HRV1-001"},
                    expected_membership_sha256=membership_sha256,
                )

    def test_execution_metadata_accepts_actual_m2_layout(self) -> None:
        membership_sha256 = "a" * 64
        shared = {
            "status": "COMPLETE",
            "retained_n": 61,
            "prediction_n": 61,
            "train_n": 1209,
            "retained_membership_sha256": membership_sha256,
            "human_labels_used_for_training_tuning_or_prediction": False,
        }
        m1 = {
            **shared,
            "method_id": "M1",
            "config_sha256": ANALYSIS.EXPECTED_SUPERVISED_CONFIG_SHA256["M1"],
            "fixed_hyperparameters": {
                "min_df": 2,
                "C": 1.0,
                "class_weight": None,
                "global_threshold": 0.25,
            },
        }
        m2 = {
            **shared,
            "method_id": "M2",
            "config_sha256": ANALYSIS.EXPECTED_SUPERVISED_CONFIG_SHA256["M2"],
            "fixed_hyperparameters": {
                "learning_rate": 3e-5,
                "weight_decay": 0.01,
                "epochs": 6,
                "global_threshold": 0.20,
            },
            "technical_execution": {
                "max_length": 2048,
                "physical_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "effective_train_batch_size": 16,
                "gradient_checkpointing": True,
                "mixed_precision_dtype": "bfloat16",
                "gradient_scaler_enabled": False,
                "adamw_foreach": False,
                "pad_to_multiple_of": 64,
            },
            "validation_run": False,
            "threshold_search": False,
        }
        diagnostics = {
            method: {
                "status": "COMPLETE",
                "method": method,
                "evaluation": "B",
                "expected_cases": 61,
                "successful_predictions": 61,
                "new_api_request_successes": 61,
                "reused_identical_requests": 0,
                "unresolved_failures": 0,
                "missing_unattempted": 0,
                "retained_membership_sha256": membership_sha256,
                "store": False,
                "human_or_silver_labels_sent_to_model": False,
                "prompt_sha256": ANALYSIS.EXPECTED_LLM_PROMPT_SHA256[method],
                "schema_sha256": ANALYSIS.EXPECTED_LLM_SCHEMA_SHA256,
                "model": "gpt-5.6-luna",
                "demo_overlap_ranks": [],
                "demo_bank_id": "A1" if method == "M4" else None,
            }
            for method in ("M3", "M4")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_paths = {
                method: root / f"{method.lower()}_predictions.jsonl"
                for method in ANALYSIS.METHODS
            }
            for method, path in prediction_paths.items():
                path.write_text(f'{{"method":"{method}"}}\n', encoding="utf-8")
            m1["prediction_path"] = str(prediction_paths["M1"])
            m1["prediction_sha256"] = ANALYSIS._sha256_file(prediction_paths["M1"])
            m2["prediction_path"] = str(prediction_paths["M2"])
            m2["prediction_sha256"] = ANALYSIS._sha256_file(prediction_paths["M2"])
            for method in ("M3", "M4"):
                diagnostics[method]["prediction_file"] = str(prediction_paths[method])
                diagnostics[method]["prediction_file_sha256"] = ANALYSIS._sha256_file(
                    prediction_paths[method]
                )
            payloads = {"m1": m1, "m2": m2, **{key.lower(): value for key, value in diagnostics.items()}}
            paths = {}
            for name, payload in payloads.items():
                paths[name] = root / f"{name}.json"
                paths[name].write_text(json.dumps(payload), encoding="utf-8")
            observed = ANALYSIS.validate_execution_metadata(
                paths["m1"],
                paths["m2"],
                paths["m3"],
                paths["m4"],
                retained_n=61,
                expected_m4_n=61,
                demo_overlap_ids=set(),
                demo_overlap_ranks=set(),
                prediction_paths=prediction_paths,
                expected_membership_sha256=membership_sha256,
                expected_m4_demo_bank_id="A1",
            )
            prediction_paths["M1"].write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ANALYSIS.EvaluationBAnalysisError, "prediction hash"
            ):
                ANALYSIS.validate_execution_metadata(
                    paths["m1"],
                    paths["m2"],
                    paths["m3"],
                    paths["m4"],
                    retained_n=61,
                    expected_m4_n=61,
                    demo_overlap_ids=set(),
                    demo_overlap_ranks=set(),
                    prediction_paths=prediction_paths,
                    expected_membership_sha256=membership_sha256,
                    expected_m4_demo_bank_id="A1",
                )
        self.assertEqual(observed["M2"]["technical_execution"]["max_length"], 2048)

    def test_llm_prediction_loader_binds_frozen_request_provenance(self) -> None:
        case = human_case("HRV1-001", 1, acts=("ACT_RECRUITMENT",))
        membership_sha256 = "a" * 64
        demo_sha256 = "b" * 64
        row = {
            "method": "M4",
            "evaluation": "B",
            "reliability_case_id": case.reliability_case_id,
            "search_rank": case.search_rank,
            "canonical_url": case.canonical_url,
            "fact_summary": case.fact_summary,
            "input_sha256": case.input_sha256,
            "retained_membership_sha256": membership_sha256,
            "status": "SUCCESS_VALIDATED",
            "predicted_labels": ["ACT_RECRUITMENT"],
            "human_or_silver_labels_sent_to_model": False,
            "store": False,
            "prompt_sha256": ANALYSIS.EXPECTED_LLM_PROMPT_SHA256["M4"],
            "schema_sha256": ANALYSIS.EXPECTED_LLM_SCHEMA_SHA256,
            "effective_requested_model_id": "gpt-5.6-luna",
            "request_sha256": "c" * 64,
            "builder_payload_sha256": "d" * 64,
            "builder_metadata_sha256": "e" * 64,
            "api_request_issued_for_evaluation_b": True,
            "reuse_status": "NEW_EVALUATION_B_REQUEST",
            "demo_bank_id": "A1",
            "demo_bank_membership_sha256": demo_sha256,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m4.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = ANALYSIS.load_predictions(
                "M4",
                path,
                {case.reliability_case_id: case},
                {case.reliability_case_id},
                expected_membership_sha256=membership_sha256,
                expected_m4_demo_bank_id="A1",
                expected_m4_demo_membership_sha256=demo_sha256,
            )
            self.assertEqual(set(loaded), {case.reliability_case_id})
            row["prompt_sha256"] = "0" * 64
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ANALYSIS.EvaluationBAnalysisError, "prompt hash"):
                ANALYSIS.load_predictions(
                    "M4",
                    path,
                    {case.reliability_case_id: case},
                    {case.reliability_case_id},
                    expected_membership_sha256=membership_sha256,
                    expected_m4_demo_bank_id="A1",
                    expected_m4_demo_membership_sha256=demo_sha256,
                )


if __name__ == "__main__":
    unittest.main()
