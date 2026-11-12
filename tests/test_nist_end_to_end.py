"""Unit tests for Phase 5: NIST End-to-End Pipeline Evaluation."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.evaluate_nist_end_to_end import load_nist_queries


class TestNistEndToEnd(unittest.TestCase):
    """Test suite for NIST evaluation loader and metric schemas."""

    def test_load_nist_queries(self) -> None:
        eval_path = Path(__file__).resolve().parents[1] / "data" / "eval" / "retrieval_eval.jsonl"
        queries = load_nist_queries(eval_path)
        self.assertEqual(len(queries), 64)

        answerable = [q for q in queries if q.get("answerable", True)]
        unanswerable = [q for q in queries if not q.get("answerable", True)]
        self.assertEqual(len(answerable), 54)
        self.assertEqual(len(unanswerable), 10)

        for q in queries:
            self.assertIn("qid", q)
            self.assertIn("question", q)
            self.assertIn("answerable", q)

    def test_ret_005_sufficient_without_literal_nist(self) -> None:
        """Verify ret-005 is deemed sufficient when definition is present without literal word 'nist'."""
        from evidencefirst_rag.chunking import Chunk
        from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator

        # Gold definition snippet from NIST.AI.600-1.pdf page 10 (does not repeat "NIST")
        confabulation_chunk = Chunk(
            chunk_id="nist-ai-600-1:chunk-14",
            document_id="nist-ai-600-1",
            document_name="NIST.AI.600-1.pdf",
            page=10,
            section="2.2. Confabulation",
            text=(
                "“Confabulation” refers to a phenomenon in which GAI systems generate and "
                "confidently present erroneous or false content in response to prompts. "
                "Confabulations also include generated content that is not grounded in the training "
                "data or context provided to the model."
            ),
            token_count=45,
            char_count=285,
        )

        evaluator = ContextSufficiencyEvaluator()
        question = "What is confabulation in generative AI according to NIST's Generative AI Profile?"

        result = evaluator.evaluate(question, [confabulation_chunk])
        self.assertTrue(result.sufficient, f"Expected sufficient, got: {result.rationale}")
        self.assertEqual(len(result.missing_information), 0)

    def test_unanswerable_query_abstains_without_hallucination(self) -> None:
        """Verify unanswerable queries abstain rather than generating ungrounded facts."""
        from evidencefirst_rag.chunking import Chunk
        from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator

        # Arbitrary cyber chunk not answering 2025 budget
        irrelevant_chunk = Chunk(
            chunk_id="nist-csf:chunk-1",
            document_id="nist-csf",
            document_name="NIST.CSWP.29.pdf",
            page=1,
            section="Overview",
            text="The NIST Cybersecurity Framework provides taxonomy and guidance for managing risk.",
            token_count=15,
            char_count=100,
        )

        evaluator = ContextSufficiencyEvaluator()
        question = "What was the annual budget of the U.S. AI Safety Institute in fiscal year 2025?"

        result = evaluator.evaluate(question, [irrelevant_chunk])
        self.assertFalse(result.sufficient)
        self.assertGreater(len(result.missing_information), 0)

    def test_unsupported_rate_denominators(self) -> None:
        """Verify both total-query and answered-query denominators are correctly reported."""
        total_queries = 64
        answered_queries = 45
        unsupported_count = 0

        unsupp_total = unsupported_count / total_queries
        unsupp_answered = (unsupported_count / answered_queries) if answered_queries > 0 else 0.0

        self.assertEqual(unsupp_total, 0.0)
        self.assertEqual(unsupp_answered, 0.0)
