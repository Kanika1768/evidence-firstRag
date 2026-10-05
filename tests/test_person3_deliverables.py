"""Unit tests for Person 3 / Kanika deliverables:

1. Prompt versioning and schema (sufficiency_v1.py)
2. Structured sufficiency adapter (validation, repair parsing, ContextAssessment conversion)
3. B1 Sufficiency-Aware baseline execution (one-shot abstention, zero recovery)
4. Monotonic latency tracking (mean and median)
5. Dataset-level Recovery Gain metric
6. CSV exports integrity (final_comparison.csv, per_question_comparison.csv, sufficiency_metrics.csv)
7. Plot outputs (accuracy_coverage_curve, confusion_matrix PNG and SVG)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from agentic_rag.adapters import StructuredSufficiencyResult
from agentic_rag.contracts import (
    AnswerStatus,
    AnswerabilityLabel,
    ContextAssessment,
    ContextStatus,
    Corpus,
    Snippet,
)
from agentic_rag.document_index import build_local_index
from agentic_rag.document_loader import LoadedDocument
from agentic_rag.generic_pipeline import build_generic_components
from agentic_rag.prompts import (
    SUFFICIENCY_PROMPT_VERSION,
    build_sufficiency_prompt,
    get_sufficiency_json_schema,
    get_sufficiency_system_prompt,
)
from examples.plot_accuracy_coverage import generate_accuracy_coverage_plots
from examples.plot_confusion_matrix import generate_confusion_matrix_plots
from examples.run_evaluation import (
    B0ConventionalRAG,
    B1SufficiencyAwareRAG,
    ControlledRecoveryRewriter,
    EvaluationRunRecord,
    compute_aggregate,
    compute_recovery_gain,
    run_evaluation_suite,
)
from examples.run_sufficiency_benchmark import export_sufficiency_metrics_csv


class TestPerson3Deliverables(unittest.TestCase):
    """Test suite verifying all Person 3 evaluation and adapter components."""

    def setUp(self) -> None:
        self.sample_docs = (
            LoadedDocument(
                document_id="company",
                document_name="company.txt",
                source_path="company.txt",
                page=1,
                text="NovaTech is a software company founded by Priya Mehta. NovaTech develops analytics software.",
            ),
            LoadedDocument(
                document_id="history",
                document_name="history.txt",
                source_path="history.txt",
                page=1,
                text="NovaTech was founded in 2019 in San Francisco.",
            ),
            LoadedDocument(
                document_id="finance",
                document_name="finance.txt",
                source_path="finance.txt",
                page=1,
                text="NovaTech reported annual revenue of 42 million dollars in 2025.",
            ),
        )
        self.corpora = tuple(
            Corpus(id=doc.document_name.split(".")[0], description=doc.text)
            for doc in self.sample_docs
        )
        self.retriever = build_local_index(self.sample_docs, chunk_size=400, overlap=50, per_query_limit=3)
        self.components = build_generic_components()

    # -----------------------------------------------------------------------
    # 1. Prompt Versioning & Schema Tests
    # -----------------------------------------------------------------------
    def test_prompt_versioning_and_templates(self) -> None:
        self.assertEqual(SUFFICIENCY_PROMPT_VERSION, "v1.0")
        sys_prompt = get_sufficiency_system_prompt()
        self.assertIn("diligent reader", sys_prompt.lower())
        self.assertIn("missing_information", sys_prompt)

        schema = get_sufficiency_json_schema()
        self.assertEqual(schema["title"], "StructuredSufficiencyResult")
        self.assertIn("sufficient", schema["properties"])
        self.assertIn("probability", schema["properties"])
        self.assertIn("missing_information", schema["properties"])
        self.assertIn("rationale", schema["properties"])

        prompt = build_sufficiency_prompt(
            question="Where was Michael Jack born?",
            context="Michael Jack was born in Folkestone in 1946.",
            required_facts=["birthplace of Michael Jack"],
        )
        self.assertIn("Where was Michael Jack born?", prompt)
        self.assertIn("Folkestone", prompt)
        self.assertIn("birthplace of Michael Jack", prompt)

    # -----------------------------------------------------------------------
    # 2. Structured Sufficiency Adapter Tests
    # -----------------------------------------------------------------------
    def test_structured_adapter_validation_and_serialization(self) -> None:
        res = StructuredSufficiencyResult(
            sufficient=True,
            probability=0.95,
            missing_information=[],
            rationale="Context directly answers question.",
        )
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 0.95)

        # Serialization to and from dict
        d = res.to_dict()
        self.assertEqual(d["sufficient"], True)
        self.assertEqual(d["probability"], 0.95)

        res2 = StructuredSufficiencyResult.from_dict(d)
        self.assertEqual(res, res2)

        # Validation bounds
        with self.assertRaises(ValueError):
            StructuredSufficiencyResult(sufficient=True, probability=1.5, rationale="test")
        with self.assertRaises(TypeError):
            StructuredSufficiencyResult(sufficient="yes", probability=0.5, rationale="test")  # type: ignore

    def test_structured_adapter_bidirectional_conversion(self) -> None:
        assessment = ContextAssessment(
            status=ContextStatus.INSUFFICIENT,
            sufficiency_score=0.25,
            missing_facts=("founding year", "headquarters"),
            reason="Missing founding year.",
        )
        structured = StructuredSufficiencyResult.from_context_assessment(assessment)
        self.assertFalse(structured.sufficient)
        self.assertEqual(structured.probability, 0.25)
        self.assertEqual(structured.missing_information, ["founding year", "headquarters"])
        self.assertEqual(structured.rationale, "Missing founding year.")

        converted_back = structured.to_context_assessment()
        self.assertEqual(converted_back.status, ContextStatus.INSUFFICIENT)
        self.assertEqual(converted_back.sufficiency_score, 0.25)
        self.assertEqual(converted_back.missing_facts, ("founding year", "headquarters"))
        self.assertEqual(converted_back.reason, "Missing founding year.")

    def test_structured_adapter_repair_fallback_parsing(self) -> None:
        # Markdown fenced code block with single quotes and trailing comma
        dirty_json = """
        Here is the evaluation result:
        ```json
        {
            'sufficient': true,
            'probability': 0.88,
            'missing_information': [],
            'rationale': 'All facts supported.',
        }
        ```
        """
        parsed = StructuredSufficiencyResult.from_json(dirty_json)
        self.assertTrue(parsed.sufficient)
        self.assertEqual(parsed.probability, 0.88)
        self.assertEqual(parsed.rationale, "All facts supported.")

        # Partial non-JSON string fallback
        raw_garbage = "I think the context is not sufficient because the CEO is missing."
        fallback = StructuredSufficiencyResult.from_json(raw_garbage)
        self.assertFalse(fallback.sufficient)
        self.assertEqual(fallback.probability, 0.0)
        self.assertIn("Failed to parse", fallback.rationale)

    # -----------------------------------------------------------------------
    # 3. B1 Sufficiency-Aware Baseline Tests
    # -----------------------------------------------------------------------
    def test_b1_baseline_execution_and_abstention(self) -> None:
        b1 = B1SufficiencyAwareRAG(
            retriever=self.retriever,
            planner=self.components["planner"],
            rewriter=self.components["rewriter"],
            judge=self.components["judge"],
            drafter=self.components["drafter"],
            synthesizer=self.components["synthesizer"],
        )

        # Sufficient question: should answer in 1 iteration
        res_suff = b1.run("What does NovaTech develop?", self.corpora)
        self.assertEqual(len(res_suff.iterations), 1)
        self.assertEqual(res_suff.answer.status, AnswerStatus.ANSWERED)
        self.assertIn("analytics software", res_suff.answer.answer.lower())

        # Unanswerable question: should selectively abstain in 1 iteration (NO recovery)
        res_unans = b1.run("Who is the CEO of NovaTech?", self.corpora)
        self.assertEqual(len(res_unans.iterations), 1)
        self.assertEqual(res_unans.answer.status, AnswerStatus.UNANSWERABLE)
        self.assertEqual(res_unans.answer.answer, "No grounded answer is available from the retrieved context.")

    def test_b0_baseline_unsupported_answering(self) -> None:
        b0 = B0ConventionalRAG(self.retriever)
        res = b0.run("Who is the CEO of NovaTech?", self.corpora)
        # B0 always answers, even on unanswerable questions
        self.assertEqual(len(res.iterations), 1)
        self.assertEqual(res.answer.status, AnswerStatus.ANSWERED)
        self.assertTrue(bool(res.answer.answer))

    # -----------------------------------------------------------------------
    # 4. Latency Tracking & Aggregation Tests
    # -----------------------------------------------------------------------
    def test_latency_tracking_computation(self) -> None:
        from examples.run_evaluation import EvaluationTestCase
        tc = EvaluationTestCase("Q1", "test", AnswerabilityLabel.SUFFICIENT)
        records = [
            EvaluationRunRecord(
                system_name="B0",
                test_case=tc,
                answer=None,  # type: ignore
                iterations=1,
                snippets=(),
                is_correct=True,
                is_grounded=True,
                is_unsupported=False,
                is_abstained=False,
                matched_expected=True,
                latency_ms=10.0,
            ),
            EvaluationRunRecord(
                system_name="B0",
                test_case=tc,
                answer=None,  # type: ignore
                iterations=1,
                snippets=(),
                is_correct=True,
                is_grounded=True,
                is_unsupported=False,
                is_abstained=False,
                matched_expected=True,
                latency_ms=20.0,
            ),
        ]
        agg = compute_aggregate("B0", records)
        self.assertEqual(agg.mean_latency_ms, 15.0)
        self.assertEqual(agg.median_latency_ms, 15.0)

    # -----------------------------------------------------------------------
    # 5. Recovery Gain Metric Tests
    # -----------------------------------------------------------------------
    def test_recovery_gain_calculation(self) -> None:
        from examples.run_evaluation import EvaluationTestCase
        tc1 = EvaluationTestCase("Q1", "test1", AnswerabilityLabel.SUFFICIENT)
        tc2 = EvaluationTestCase("Q2", "test2", AnswerabilityLabel.SUFFICIENT)

        # B1: fails Q1 (e.g. abstained), correct on Q2
        b1_recs = [
            EvaluationRunRecord("B1", tc1, None, 1, (), False, False, False, True, False, 1.0),  # type: ignore
            EvaluationRunRecord("B1", tc2, None, 1, (), True, True, False, False, True, 1.0),  # type: ignore
        ]
        # E1: recovers Q1 correctly, correct on Q2
        e1_recs = [
            EvaluationRunRecord("E1", tc1, None, 2, (), True, True, False, False, True, 2.0),  # type: ignore
            EvaluationRunRecord("E1", tc2, None, 1, (), True, True, False, False, True, 1.0),  # type: ignore
        ]

        gain_info = compute_recovery_gain(b1_recs, e1_recs)
        self.assertEqual(gain_info["newly_correct"], 1)
        self.assertEqual(gain_info["newly_unsupported"], 0)
        self.assertEqual(gain_info["recovery_gain"], 1)

    # -----------------------------------------------------------------------
    # 6. CSV Export Integrity Tests
    # -----------------------------------------------------------------------
    def test_csv_exports_exist_and_contain_required_columns(self) -> None:
        # Run evaluation suite to generate CSVs
        run_evaluation_suite()

        final_csv = Path("results/final_comparison.csv")
        per_q_csv = Path("results/per_question_comparison.csv")
        suff_csv = Path("results/sufficiency_metrics.csv")

        self.assertTrue(final_csv.exists())
        self.assertTrue(per_q_csv.exists())

        # Export sufficiency metrics to ensure file is current
        export_sufficiency_metrics_csv({
            "total": 80.0,
            "tp": 12.0,
            "tn": 39.0,
            "fp": 1.0,
            "fn": 28.0,
            "accuracy": 0.6375,
            "precision": 0.9231,
            "recall": 0.3000,
            "f1": 0.4528,
        })
        self.assertTrue(suff_csv.exists())

        # Check final_comparison.csv headers
        with open(final_csv, encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.assertIn("system_name", headers)
            self.assertIn("mean_latency_ms", headers)
            self.assertIn("median_latency_ms", headers)
            self.assertIn("recovery_gain", headers)

        # Check sufficiency_metrics.csv AUROC note
        with open(suff_csv, encoding="utf-8") as f:
            content = f.read()
            self.assertIn("AUROC,N/A", content)

    # -----------------------------------------------------------------------
    # 7. Plot Output Integrity Tests
    # -----------------------------------------------------------------------
    def test_plot_generation_and_headers(self) -> None:
        p1_png, p1_svg = generate_accuracy_coverage_plots()
        self.assertTrue(p1_png.exists())
        self.assertTrue(p1_svg.exists())

        p2_png, p2_svg = generate_confusion_matrix_plots()
        self.assertTrue(p2_png.exists())
        self.assertTrue(p2_svg.exists())

        # Verify PNG magic bytes (\x89PNG\r\n\x1a\n)
        png_magic = b"\x89PNG\r\n\x1a\n"
        self.assertEqual(p1_png.read_bytes()[:8], png_magic)
        self.assertEqual(p2_png.read_bytes()[:8], png_magic)


if __name__ == "__main__":
    unittest.main()
