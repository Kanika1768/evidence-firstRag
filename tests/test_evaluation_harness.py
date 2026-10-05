"""Unit tests for the B0 vs E1 evaluation harness and metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from agentic_rag.contracts import (
    AnswerStatus,
    AnswerabilityLabel,
    ContextStatus,
    Corpus,
    GroundedAnswer,
    GroundedCitation,
    Snippet,
)
from agentic_rag.document_index import build_local_index
from agentic_rag.document_loader import load_documents
from agentic_rag.generic_pipeline import build_generic_components
from agentic_rag.orchestrator import AgenticRAGOrchestrator, OrchestratorConfig
from run_evaluation import (
    B0ConventionalRAG,
    ControlledRecoveryRewriter,
    EvaluationTestCase,
    compute_aggregate,
    evaluate_run_record,
)


class EvaluationHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = ROOT / "data" / "demo_documents"
        cls.documents = load_documents(cls.data_dir)
        cls.corpora = tuple(
            Corpus(
                id=Path(doc.document_name).stem,
                description=f"{doc.document_name} {doc.text[:500]}",
            )
            for doc in cls.documents
        )
        cls.retriever = build_local_index(
            cls.documents,
            chunk_size=800,
            overlap=120,
            per_query_limit=5,
        )
        cls.components = build_generic_components()
        cls.e1_orchestrator = AgenticRAGOrchestrator(
            planner=cls.components["planner"],
            rewriter=cls.components["rewriter"],
            retriever=cls.retriever,
            drafter=cls.components["drafter"],
            judge=cls.components["judge"],
            synthesizer=cls.components["synthesizer"],
            config=OrchestratorConfig(max_iterations=2),
        )

    def test_b0_baseline_is_oneshot_and_never_abstains(self):
        b0 = B0ConventionalRAG(self.retriever)
        result = b0.run("Who is the CEO of NovaTech?", self.corpora)

        self.assertEqual(1, len(result.iterations))
        self.assertEqual(AnswerStatus.ANSWERED, result.answer.status)
        # B0 returns whatever text matched 'NovaTech' rather than abstaining
        self.assertTrue(len(result.answer.answer) > 0)

    def test_e1_evidencefirst_abstains_on_unanswerable_question(self):
        result = self.e1_orchestrator.run("Who is the CEO of NovaTech?", self.corpora)

        self.assertEqual(AnswerStatus.UNANSWERABLE, result.answer.status)
        self.assertEqual(0.0, result.answer.sufficiency_score)
        self.assertTrue(len(result.answer.missing_facts) > 0)
        self.assertIn("Who is the CEO of NovaTech", result.answer.missing_facts[0])

    def test_e1_evidencefirst_multi_corpus_fact_grounding(self):
        result = self.e1_orchestrator.run(
            "Who founded NovaTech and when was it founded?",
            self.corpora,
        )

        status_str = result.answer.status.value if hasattr(result.answer.status, "value") else str(result.answer.status)
        self.assertEqual("answered", status_str)
        self.assertEqual(1.0, result.answer.sufficiency_score)
        answer_lower = result.answer.answer.lower()
        self.assertIn("priya mehta", answer_lower)
        self.assertIn("2019", answer_lower)
        self.assertGreaterEqual(len(result.answer.citations), 2)

    def test_controlled_recovery_demonstration_transitions_insufficient_to_sufficient(self):
        rewriter = ControlledRecoveryRewriter(self.components["rewriter"])
        orchestrator = AgenticRAGOrchestrator(
            planner=self.components["planner"],
            rewriter=rewriter,
            retriever=self.retriever,
            drafter=self.components["drafter"],
            judge=self.components["judge"],
            synthesizer=self.components["synthesizer"],
            config=OrchestratorConfig(max_iterations=2),
        )

        result = orchestrator.run("What is NovaTech's annual revenue?", self.corpora)

        self.assertEqual(2, len(result.iterations))
        # Iteration 1 was insufficient
        self.assertEqual(ContextStatus.INSUFFICIENT, result.iterations[0].assessment.status)
        # Iteration 2 recovered and became sufficient
        self.assertEqual(ContextStatus.SUFFICIENT, result.iterations[1].assessment.status)
        status_str = result.answer.status.value if hasattr(result.answer.status, "value") else str(result.answer.status)
        self.assertEqual("answered", status_str)
        self.assertIn("42 million", result.answer.answer.lower())

    def test_metrics_computation(self):
        test_case_s = EvaluationTestCase(
            id="T1",
            question="What does NovaTech develop?",
            expected_label=AnswerabilityLabel.SUFFICIENT,
            expected_terms=("analytics software",),
        )
        test_case_u = EvaluationTestCase(
            id="T2",
            question="Who is the CEO of NovaTech?",
            expected_label=AnswerabilityLabel.UNANSWERABLE,
            expected_terms=(),
        )

        b0_res_s = B0ConventionalRAG(self.retriever).run(test_case_s.question, self.corpora)
        b0_res_u = B0ConventionalRAG(self.retriever).run(test_case_u.question, self.corpora)

        rec_s = evaluate_run_record(test_case_s, b0_res_s, "B0")
        rec_u = evaluate_run_record(test_case_u, b0_res_u, "B0")

        self.assertTrue(rec_s.is_correct)
        self.assertFalse(rec_u.is_correct)
        self.assertTrue(rec_u.is_unsupported)
        self.assertFalse(rec_u.is_abstained)

        aggregate = compute_aggregate("B0", (rec_s, rec_u))
        self.assertEqual(2, aggregate.total_questions)
        self.assertEqual(1, aggregate.correct_count)
        self.assertEqual(0.5, aggregate.accuracy)
        self.assertEqual(0.5, aggregate.unsupported_rate)
        self.assertEqual(0.0, aggregate.abstention_rate)


if __name__ == "__main__":
    unittest.main()
