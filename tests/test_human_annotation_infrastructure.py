"""Unit tests for the Human Validation Infrastructure (Phase 1)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from evidencefirst_rag.annotation_tools import (
    ReviewerAnnotation,
    adjudicate_and_promote_gold,
    calculate_inter_annotator_agreement,
    export_blinded_annotation_package,
    generate_validation_status_summary,
    load_and_validate_reviews,
)


class TestHumanAnnotationInfrastructure(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark_path = ROOT / "data" / "evaluation" / "benchmark.jsonl"

    def test_status_summary_reports_genuine_pending_state(self) -> None:
        summary = generate_validation_status_summary(self.benchmark_path)
        self.assertEqual(summary["total_instances"], 80)
        self.assertEqual(summary["verified_human_gold_count"], 0)
        self.assertEqual(summary["pending_human_validation_count"], 80)
        self.assertEqual(summary["validation_completion_pct"], 0.0)
        self.assertEqual(summary["status"], "PENDING HUMAN REVIEW")

    def test_blinded_package_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_p, csv_p = export_blinded_annotation_package(
                self.benchmark_path, Path(tmpdir), "test_package"
            )
            self.assertTrue(jsonl_p.exists())
            self.assertTrue(csv_p.exists())

            # Verify JSONL lines
            with open(jsonl_p, encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(records), 80)
            for r in records:
                self.assertIn("id", r)
                self.assertIn("original_question", r)
                self.assertIn("context", r)
                self.assertIn("human_decision", r)
                self.assertEqual(r["human_decision"], "")
                self.assertEqual(r["human_rationale"], "")
                # Critical blinding requirement: candidate_label and curator_rationale must NOT be exposed
                self.assertNotIn("candidate_label", r)
                self.assertNotIn("curator_candidate_label", r)
                self.assertNotIn("curator_rationale", r)

    def test_review_validation_enforces_schema(self) -> None:
        valid_ann = ReviewerAnnotation(
            id="rec-01",
            decision="SUFFICIENT",
            rationale="Context gives the complete answer.",
            annotator_id="rev-1",
        )
        valid_ann.validate()

        invalid_decision = ReviewerAnnotation(
            id="rec-02",
            decision="MAYBE",
            rationale="Not sure",
            annotator_id="rev-1",
        )
        with self.assertRaises(ValueError):
            invalid_decision.validate()

        empty_rationale = ReviewerAnnotation(
            id="rec-03",
            decision="INSUFFICIENT",
            rationale="",
            annotator_id="rev-1",
        )
        with self.assertRaises(ValueError):
            empty_rationale.validate()

    def test_inter_annotator_agreement_calculation(self) -> None:
        # Perfect agreement
        rev_a = {
            f"rec-{i}": ReviewerAnnotation(f"rec-{i}", "SUFFICIENT", "Clear", "A") for i in range(10)
        }
        rev_b = {
            f"rec-{i}": ReviewerAnnotation(f"rec-{i}", "SUFFICIENT", "Clear", "B") for i in range(10)
        }
        report = calculate_inter_annotator_agreement(rev_a, rev_b)
        self.assertEqual(report.observed_agreement, 1.0)
        self.assertEqual(report.cohens_kappa, 1.0)
        self.assertEqual(len(report.discrepancy_ids), 0)

        # Mixed agreement with known mismatch
        rev_b["rec-0"] = ReviewerAnnotation("rec-0", "INSUFFICIENT", "Missing detail", "B")
        report2 = calculate_inter_annotator_agreement(rev_a, rev_b)
        self.assertEqual(report2.agreed_count, 9)
        self.assertEqual(report2.disagreed_count, 1)
        self.assertEqual(report2.discrepancy_ids, ["rec-0"])
        self.assertLess(report2.cohens_kappa, 1.0)

    def test_adjudicate_and_promote_gold_preserves_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_bench = Path(tmpdir) / "bench.jsonl"
            sample_data = [
                {
                    "id": "item-1",
                    "candidate_label": "SUFFICIENT",
                    "gold_label": "PENDING",
                    "original_question": "Q1?",
                },
                {
                    "id": "item-2",
                    "candidate_label": "INSUFFICIENT",
                    "gold_label": "PENDING",
                    "original_question": "Q2?",
                },
            ]
            with open(tmp_bench, "w", encoding="utf-8") as f:
                for d in sample_data:
                    f.write(json.dumps(d) + "\n")

            promoted = adjudicate_and_promote_gold(
                tmp_bench,
                {"item-1": "SUFFICIENT", "item-2": "INSUFFICIENT"},
                tmp_bench,
            )
            self.assertEqual(promoted, 2)

            with open(tmp_bench, encoding="utf-8") as f:
                updated = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(updated[0]["gold_label"], "SUFFICIENT")
            self.assertEqual(updated[0]["candidate_label"], "SUFFICIENT")
            self.assertEqual(updated[1]["gold_label"], "INSUFFICIENT")
            self.assertEqual(updated[1]["candidate_label"], "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
