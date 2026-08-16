#!/usr/bin/env python3
"""CLI for Evaluation B reference preparation and later guarded analyses.

There is deliberately no case-selection command.  ``freeze-reference`` uses
the immutable completed single-reviewer file and does not run a model or API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation_b import (
    EvaluationBError,
    build_disagreement_queue,
    build_human_gold,
    compare_silver_to_human,
    compute_reviewer_agreement,
    evaluate_human_gold_predictions,
    generate_frozen_reliability_provenance,
    load_csv_rows,
    load_jsonl_rows,
    qc_annotations,
    write_qc_reports,
)
from evaluation_b_reference import build_single_reviewer_reference


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELIABILITY_SAMPLE = REPO_ROOT / "data/annotations/reliability_sample_100.csv"
DEFAULT_A1_SPLIT = REPO_ROOT / "data/splits/a1_iid_split_final_v1.csv"
DEFAULT_A2_SPLIT = REPO_ROOT / "data/splits/a2_jurisdiction_folds_final_v1.csv"
DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs/analysis/evaluation_b"
DEFAULT_PROVENANCE = DEFAULT_ANALYSIS_DIR / "reliability_case_experiment_provenance.csv"
DEFAULT_HUMAN_SOURCE = REPO_ROOT / "data/annotations/reviewer_annotation_template.csv"
DEFAULT_SOURCE_MANIFEST = DEFAULT_ANALYSIS_DIR / "human_annotation_source_manifest.json"
DEFAULT_HUMAN_QC_REPORT = DEFAULT_ANALYSIS_DIR / "human_annotation_qc_report.csv"
DEFAULT_HUMAN_QC_SUMMARY = DEFAULT_ANALYSIS_DIR / "human_annotation_qc_summary.json"
DEFAULT_HUMAN_REFERENCE = REPO_ROOT / "data/annotations/human_grounded_reference_v1.csv"
DEFAULT_HUMAN_EXCLUSIONS = REPO_ROOT / "data/annotations/human_grounded_reference_exclusions_v1.csv"
DEFAULT_HUMAN_MEMBERSHIP = DEFAULT_ANALYSIS_DIR / "human_grounded_reference_membership_v1.csv"
DEFAULT_EVAL_B_FREEZE = DEFAULT_ANALYSIS_DIR / "eval_b_membership_manifest.json"


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationBError(f"Refusing to write empty derived artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_rows(path) if path.suffix.lower() == ".jsonl" else load_csv_rows(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    provenance = subparsers.add_parser(
        "provenance", help="Generate the label-free reliability/split provenance table"
    )
    provenance.add_argument("--reliability-sample", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    provenance.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    provenance.add_argument("--a2-split", type=Path, default=DEFAULT_A2_SPLIT)
    provenance.add_argument("--output", type=Path, default=DEFAULT_PROVENANCE)

    freeze = subparsers.add_parser(
        "freeze-reference",
        help="QC and freeze the immutable Done?-gated single-reviewer reference",
    )
    freeze.add_argument("--source", type=Path, default=DEFAULT_HUMAN_SOURCE)
    freeze.add_argument("--context", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    freeze.add_argument("--a1-split", type=Path, default=DEFAULT_A1_SPLIT)
    freeze.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    freeze.add_argument("--qc-report", type=Path, default=DEFAULT_HUMAN_QC_REPORT)
    freeze.add_argument("--qc-summary", type=Path, default=DEFAULT_HUMAN_QC_SUMMARY)
    freeze.add_argument("--reference", type=Path, default=DEFAULT_HUMAN_REFERENCE)
    freeze.add_argument("--exclusions", type=Path, default=DEFAULT_HUMAN_EXCLUSIONS)
    freeze.add_argument("--membership", type=Path, default=DEFAULT_HUMAN_MEMBERSHIP)
    freeze.add_argument("--freeze-manifest", type=Path, default=DEFAULT_EVAL_B_FREEZE)

    qc = subparsers.add_parser("qc", help="Validate one future reviewer file without modifying it")
    qc.add_argument("--annotations", type=Path, required=True)
    qc.add_argument("--expected-cases", type=Path)
    qc.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)

    agreement = subparsers.add_parser("agreement", help="Compare two locked reviewer files")
    agreement.add_argument("--reviewer-a", type=Path, required=True)
    agreement.add_argument("--reviewer-b", type=Path, required=True)
    agreement.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)

    queue = subparsers.add_parser("queue", help="Create a human-only disagreement queue")
    queue.add_argument("--reviewer-a", type=Path, required=True)
    queue.add_argument("--reviewer-b", type=Path, required=True)
    queue.add_argument("--case-context", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    queue.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data/annotations/adjudication_queue.csv"
    )

    gold = subparsers.add_parser("gold", help="Build human gold after explicit adjudication")
    gold.add_argument("--reviewer-a", type=Path, required=True)
    gold.add_argument("--reviewer-b", type=Path, required=True)
    gold.add_argument("--adjudication", type=Path, required=True)
    gold.add_argument("--case-context", type=Path, default=DEFAULT_RELIABILITY_SAMPLE)
    gold.add_argument("--output", type=Path, default=REPO_ROOT / "data/annotations/human_gold_v1.csv")

    comparison = subparsers.add_parser(
        "silver-compare", help="Compare future human gold with an explicitly supplied silver artifact"
    )
    comparison.add_argument("--human-gold", type=Path, required=True)
    comparison.add_argument("--silver-reference", type=Path, required=True)
    comparison.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)

    evaluation = subparsers.add_parser(
        "evaluate", help="Evaluate explicitly supplied predictions against future human gold"
    )
    evaluation.add_argument("--human-gold", type=Path, required=True)
    evaluation.add_argument("--predictions", type=Path, required=True)
    evaluation.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "provenance":
        rows = generate_frozen_reliability_provenance(
            reliability_sample_path=args.reliability_sample,
            a1_split_path=args.a1_split,
            a2_split_path=args.a2_split,
            output_path=args.output,
        )
        print(f"Wrote {len(rows)} label-free provenance rows to {args.output}")
        return 0

    if args.command == "freeze-reference":
        result = build_single_reviewer_reference(
            repo_root=REPO_ROOT,
            source_path=args.source,
            context_path=args.context,
            a1_split_path=args.a1_split,
            source_manifest_path=args.source_manifest,
            qc_report_path=args.qc_report,
            qc_summary_path=args.qc_summary,
            reference_path=args.reference,
            exclusions_path=args.exclusions,
            membership_path=args.membership,
            freeze_manifest_path=args.freeze_manifest,
        )
        print(json.dumps(result.qc_summary, indent=2, sort_keys=True))
        print(f"Frozen retained reference rows: {len(result.reference_rows)}")
        return 0

    if args.command == "qc":
        reviewer_rows = load_csv_rows(args.annotations)
        expected_rows = load_csv_rows(args.expected_cases) if args.expected_cases else None
        expected_ids = (
            [row["reliability_case_id"] for row in expected_rows]
            if expected_rows is not None
            else None
        )
        sentence_map = (
            {
                row["reliability_case_id"]: row.get("fact_summary_numbered", "")
                for row in expected_rows
            }
            if expected_rows is not None
            else None
        )
        if sentence_map is not None:
            from evaluation_b import extract_numbered_sentences

            sentence_map = {
                key: extract_numbered_sentences(value) for key, value in sentence_map.items()
            }
        result = qc_annotations(
            reviewer_rows,
            expected_case_ids=expected_ids,
            sentence_map_by_case=sentence_map,
        )
        write_qc_reports(
            result,
            machine_path=args.output_dir / "annotation_qc_report.csv",
            human_path=args.output_dir / "annotation_qc_issue_report.md",
        )
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return 0 if result.passed else 2

    if args.command == "agreement":
        result = compute_reviewer_agreement(
            load_csv_rows(args.reviewer_a), load_csv_rows(args.reviewer_b)
        )
        _atomic_csv(args.output_dir / "reviewer_agreement_summary.csv", result["summary"])
        _atomic_csv(args.output_dir / "reviewer_agreement_per_label.csv", result["per_label"])
        _atomic_csv(args.output_dir / "reviewer_agreement_confusion_matrix.csv", result["confusion_matrix"])
        _atomic_json(args.output_dir / "reviewer_agreement_metadata.json", result["metadata"])
        return 0

    if args.command == "queue":
        rows = build_disagreement_queue(
            load_csv_rows(args.reviewer_a),
            load_csv_rows(args.reviewer_b),
            case_context_rows=load_csv_rows(args.case_context),
        )
        if rows:
            _atomic_csv(args.output, rows)
        else:
            # Empty queue is a valid result but requires an explicit schema.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("reliability_case_id,disagreement_fields\n", encoding="utf-8")
        print(f"Disagreement cases: {len(rows)}")
        return 0

    if args.command == "gold":
        rows = build_human_gold(
            load_csv_rows(args.reviewer_a),
            load_csv_rows(args.reviewer_b),
            load_csv_rows(args.adjudication),
            case_context_rows=load_csv_rows(args.case_context),
        )
        _atomic_csv(args.output, rows)
        return 0

    if args.command == "silver-compare":
        result = compare_silver_to_human(
            _load_rows(args.human_gold), _load_rows(args.silver_reference)
        )
        _atomic_csv(args.output_dir / "silver_vs_human_summary.csv", result["summary"])
        _atomic_csv(args.output_dir / "silver_vs_human_per_label.csv", result["per_label"])
        _atomic_csv(args.output_dir / "silver_vs_human_case_level.csv", result["case_level"])
        _atomic_json(args.output_dir / "silver_vs_human_metadata.json", result["metadata"])
        return 0

    if args.command == "evaluate":
        result = evaluate_human_gold_predictions(
            _load_rows(args.human_gold), _load_rows(args.predictions)
        )
        aggregate = result["aggregate"]
        cpmr = result["cpmr"]
        aggregate_row = {
            "case_n": result["metadata"]["case_n"],
            "macro_f1": aggregate["macro_f1"],
            "micro_f1": aggregate["micro_f1"],
            "exact_set_accuracy": aggregate["exact_set_accuracy"],
            "example_jaccard": aggregate["example_jaccard"],
            "act_cpmr": cpmr["ACT"]["cpmr"],
            "act_mean_contained_recall": cpmr["ACT"]["mean_contained_recall"],
            "means_cpmr": cpmr["MEANS"]["cpmr"],
            "means_mean_contained_recall": cpmr["MEANS"]["mean_contained_recall"],
            "purpose_cpmr": cpmr["PURPOSE"]["cpmr"],
            "purpose_mean_contained_recall": cpmr["PURPOSE"]["mean_contained_recall"],
            "reference_term": result["metadata"]["reference_term"],
        }
        _atomic_csv(args.output_dir / "human_gold_model_results.csv", [aggregate_row])
        _atomic_json(args.output_dir / "human_gold_model_results.json", result)
        _atomic_csv(args.output_dir / "human_gold_model_per_label.csv", result["per_label"])
        _atomic_csv(args.output_dir / "human_gold_model_per_family.csv", result["per_family"])
        _atomic_csv(args.output_dir / "human_gold_answerability_strata.csv", result["answerability_strata"])
        _atomic_csv(args.output_dir / "human_gold_case_level_errors.csv", result["case_level"])
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
