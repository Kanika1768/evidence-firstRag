#!/usr/bin/env python3
"""CLI utility for managing human validation of the Sufficiency Benchmark.

Commands:
  status     Show current human validation progress and label status
  export     Export reviewer-friendly blinded CSV and JSONL packages
  validate   Validate a completed reviewer annotation file
  agreement  Calculate inter-annotator agreement (Cohen's Kappa) between 2 reviewers
  promote    Promote verified adjudicated labels into gold_label (replaces PENDING)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.annotation_tools import (
    adjudicate_and_promote_gold,
    calculate_inter_annotator_agreement,
    export_blinded_annotation_package,
    generate_validation_status_summary,
    load_and_validate_reviews,
)

DEFAULT_BENCHMARK = ROOT / "data" / "evaluation" / "benchmark.jsonl"
DEFAULT_EXPORT_DIR = ROOT / "data" / "evaluation" / "reviewer_packages"


def cmd_status(args: argparse.Namespace) -> None:
    summary = generate_validation_status_summary(args.benchmark)
    print("=" * 70)
    print("EVIDENCEFIRST RAG: HUMAN VALIDATION STATUS REPORT")
    print("=" * 70)
    print(f"Benchmark File:     {summary['benchmark_file']}")
    print(f"Total Instances:    {summary['total_instances']}")
    print(f"Human Gold Labels:  {summary['verified_human_gold_count']} (Verified)")
    print(f"Pending Reviews:    {summary['pending_human_validation_count']} ({summary['status']})")
    print(f"Completion Rate:    {summary['validation_completion_pct']}%\n")
    print("Label Breakdown:")
    for lbl, count in summary["gold_label_distribution"].items():
        print(f"  - {lbl}: {count}")
    print("=" * 70)
    print("NOTE: Gold labels strictly require independent human review.")
    print("Candidate/proxy development labels are stored separately and")
    print("must not be reported as human-validated ground truth.")
    print("=" * 70)


def cmd_export(args: argparse.Namespace) -> None:
    jsonl_p, csv_p = export_blinded_annotation_package(
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        package_name=args.name,
    )
    print("=" * 70)
    print("BLINDED REVIEWER PACKAGES EXPORTED")
    print("=" * 70)
    print(f"JSONL Task File: {jsonl_p}")
    print(f"CSV Task File:   {csv_p}")
    print("\nBlinding Assurance: Candidate labels, curator rationales, and model")
    print("predictions were omitted to ensure unbiased human annotation.")
    print("=" * 70)


def cmd_validate(args: argparse.Namespace) -> None:
    reviews = load_and_validate_reviews(args.file)
    print(f"✓ Validated {len(reviews)} annotations from {args.file}. Schema is compliant.")


def cmd_agreement(args: argparse.Namespace) -> None:
    reviews_a = load_and_validate_reviews(args.file_a)
    reviews_b = load_and_validate_reviews(args.file_b)

    records = []
    if args.benchmark.exists():
        with open(args.benchmark, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    report = calculate_inter_annotator_agreement(reviews_a, reviews_b, benchmark_records=records)
    rep_dict = report.to_dict()

    print("=" * 70)
    print("INTER-ANNOTATOR AGREEMENT REPORT (DILIGENT READER PROTOCOL)")
    print("=" * 70)
    print(f"Common Evaluated Records: {report.total_records}")
    print(f"Agreed Records:           {report.agreed_count} ({rep_dict['observed_agreement_pct']}%)")
    print(f"Discrepant Records:       {report.disagreed_count}")
    print(f"Expected Agreement (Pe):  {rep_dict['expected_agreement_pct']}%")
    print(f"Cohen's Kappa (κ):        {report.cohens_kappa:.4f}")
    if report.cohens_kappa >= 0.85:
        print("Reliability Assessment:   STRONG AGREEMENT (κ >= 0.85)")
    elif report.cohens_kappa >= 0.70:
        print("Reliability Assessment:   MODERATE AGREEMENT (0.70 <= κ < 0.85)")
    else:
        print("Reliability Assessment:   LOW AGREEMENT - REVIEWS REQUIRE ADJUDICATION")

    print("\nConfusion Matrix (Rows=Reviewer A, Cols=Reviewer B):")
    for r_lbl, cols in report.confusion_matrix.items():
        print(f"  {r_lbl:<14}: " + " | ".join(f"{c_lbl}={count}" for c_lbl, count in cols.items()))

    if report.category_agreement:
        print("\nAgreement by Reasoning Type:")
        for cat, stats in report.category_agreement.items():
            print(f"  - {cat:<26}: {stats['agreed']}/{stats['total']} ({stats['agreement_pct']}%)")

    if report.discrepancy_ids:
        print(f"\nDiscrepancies Requiring Adjudication ({len(report.discrepancy_ids)} items):")
        for did in report.discrepancy_ids[:10]:
            print(f"  - {did}: A='{reviews_a[did].decision}' vs B='{reviews_b[did].decision}'")
        if len(report.discrepancy_ids) > 10:
            print(f"  ... and {len(report.discrepancy_ids) - 10} more.")
    print("=" * 70)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, indent=2)
        print(f"Report exported to: {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EvidenceFirst Human Validation Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = subparsers.add_parser("status", help="Show validation progress")
    p_status.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)

    # export
    p_export = subparsers.add_parser("export", help="Export reviewer packages")
    p_export.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p_export.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    p_export.add_argument("--name", type=str, default="blinded_annotation_package")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate reviewer submission")
    p_val.add_argument("file", type=Path, help="CSV or JSONL review file")

    # agreement
    p_agr = subparsers.add_parser("agreement", help="Calculate Cohen's Kappa")
    p_agr.add_argument("file_a", type=Path, help="Reviewer A file")
    p_agr.add_argument("file_b", type=Path, help="Reviewer B file")
    p_agr.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    p_agr.add_argument("--output-json", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "status":
        cmd_status(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "agreement":
        cmd_agreement(args)


if __name__ == "__main__":
    main()
