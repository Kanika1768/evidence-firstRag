"""Comprehensive Unit & Integration Test Suite for EvidenceFirst RAG Team Pipeline.

Verifies all roles under the Master Plan:
- Person 1 (Avni): Ingestion, cleaning, chunking, dense FAISS, BM25, RRF fusion, reranker, ablation
- Person 2 (Darshit): Adaptive recovery controller, grounded generation, claim citation verifier, pipeline trace
- Person 3 (Kanika): Context sufficiency evaluation, selective answering policy, 105-question benchmark verification
"""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from evidencefirst_rag.chunking import Chunk, chunk_document_pages, chunk_page, count_tokens, extract_section_heading
from evidencefirst_rag.evaluation import BaselineEvaluator, BaselineQuestion, SelectiveAnsweringModel, SelectiveAnsweringPolicy
from evidencefirst_rag.generation import GroundedAnswerRecord, GroundedGenerator
from evidencefirst_rag.ingestion import DocumentPage, clean_page_text
from evidencefirst_rag.pipeline import EvidenceFirstPipeline, PipelineTrace
from evidencefirst_rag.recovery import AdaptiveRecoveryController
from evidencefirst_rag.retrieval import (
    BM25Index,
    CrossEncoderReranker,
    DenseEmbeddingModel,
    FAISSVectorIndex,
    HybridRetriever,
    RetrievalEvalCase,
    evaluate_retrieval_ablation,
    reciprocal_rank_fusion,
)
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator
from evidencefirst_rag.verifier import CitationVerifier

ROOT = Path(__file__).resolve().parents[1]


class TestEvidenceFirstIngestionAndChunking(unittest.TestCase):
    """Test Person 1 Ingestion and Chunking modules."""

    def test_clean_page_text_removes_running_headers_and_footers(self) -> None:
        raw = (
            "Header: Advanced Networks 2026\n"
            "Page 1 of 12\n\n"
            "NovaTech develops cloud-based analytics software.\n\n"
            "Copyright (c) 2026 All Rights Reserved\n"
            "4"
        )
        cleaned = clean_page_text(raw)
        self.assertIn("NovaTech develops cloud-based analytics software.", cleaned)
        self.assertNotIn("Page 1 of 12", cleaned)
        self.assertNotIn("Copyright (c)", cleaned)

    def test_chunking_with_heading_and_provenance(self) -> None:
        page = DocumentPage(
            document_id="doc_a",
            document_name="doc_a.txt",
            source_path="data/doc_a.txt",
            page=1,
            text="# Section 1: Introduction\nNovaTech is a software company founded by Priya Mehta in 2019.",
        )
        chunks, next_idx = chunk_page(page, target_tokens=550, overlap_tokens=80, chunk_index_start=0)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.document_id, "doc_a")
        self.assertEqual(c.page, 1)
        self.assertEqual(c.section, "Section 1: Introduction")
        self.assertGreater(c.token_count, 0)
        self.assertEqual(next_idx, 1)

    def test_chunk_serialization_and_deserialization(self) -> None:
        c = Chunk(
            chunk_id="test:chunk-0",
            document_id="test",
            document_name="test.txt",
            page=1,
            section="General",
            text="Testing chunk serialization.",
            token_count=4,
            char_count=28,
        )
        d = c.to_dict()
        c2 = Chunk.from_dict(d)
        self.assertEqual(c.chunk_id, c2.chunk_id)
        self.assertEqual(c.text, c2.text)


class TestEvidenceFirstHybridRetrieval(unittest.TestCase):
    """Test Person 1 Dense + BM25 + RRF + Cross-Encoder retrieval."""

    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech is a software company founded by Priya Mehta.",
                token_count=10,
                char_count=54,
            ),
            Chunk(
                chunk_id="c2",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Products",
                text="NovaTech develops cloud-based analytics software in Bengaluru.",
                token_count=10,
                char_count=62,
            ),
            Chunk(
                chunk_id="c3",
                document_id="finance",
                document_name="finance.txt",
                page=1,
                section="Revenue",
                text="NovaTech reported annual revenue of 42 million dollars in 2025.",
                token_count=11,
                char_count=63,
            ),
        ]

    def test_dense_embedding_and_faiss_search(self) -> None:
        embedder = DenseEmbeddingModel()
        vec = embedder.encode("NovaTech products")
        self.assertEqual(len(vec), 384)
        norm = sum(x * x for x in vec) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=3)

        index = FAISSVectorIndex()
        for c in self.chunks:
            index.add(c.chunk_id, embedder.encode(c.text))
        results = index.search(vec, top_k=2)
        self.assertEqual(len(results), 2)

    def test_bm25_index_search(self) -> None:
        bm25 = BM25Index()
        bm25.fit(self.chunks)
        scores = bm25.search("revenue million dollars")
        self.assertEqual(scores[0][0], "c3")
        self.assertGreater(scores[0][1], 0.0)

    def test_reciprocal_rank_fusion(self) -> None:
        dense_ranks = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
        lexical_ranks = [("c2", 5.0), ("c3", 4.0), ("c1", 1.0)]
        fused = reciprocal_rank_fusion(dense_ranks, lexical_ranks, k=60)
        # c2 is rank 2 in dense and rank 1 in lexical -> highest combined score
        self.assertEqual(fused[0][0], "c2")

    def test_hybrid_retriever_modes_and_ablation(self) -> None:
        retriever = HybridRetriever(self.chunks)
        top_chunks = retriever.retrieve("What product does NovaTech develop?", top_k=2, mode="hybrid_rerank")
        self.assertTrue(len(top_chunks) <= 2)
        self.assertEqual(top_chunks[0].chunk_id, "c2")

        eval_cases = [
            RetrievalEvalCase("What does NovaTech develop?", ["c2"]),
            RetrievalEvalCase("What is NovaTech revenue?", ["c3"]),
        ]
        ablation = evaluate_retrieval_ablation(eval_cases, retriever)
        for mode in ("dense_only", "bm25_only", "hybrid", "hybrid_rerank"):
            self.assertIn(mode, ablation)
            self.assertIn("Recall@5", ablation[mode])
            self.assertIn("MRR", ablation[mode])


class TestEvidenceFirstSufficiencyAndRecovery(unittest.TestCase):
    """Test Person 3 Sufficiency Evaluator and Person 2 Recovery Controller."""

    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="comp-1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech is a software company founded by Priya Mehta. NovaTech develops cloud-based analytics software.",
                token_count=18,
                char_count=107,
            ),
            Chunk(
                chunk_id="hist-1",
                document_id="history",
                document_name="history.txt",
                page=1,
                section="History",
                text="NovaTech was founded in 2019 in Bengaluru.",
                token_count=8,
                char_count=43,
            ),
        ]
        self.evaluator = ContextSufficiencyEvaluator()
        self.recovery = AdaptiveRecoveryController(max_recovery_attempts=1)

    def test_sufficient_evaluation_on_answerable_query(self) -> None:
        res = self.evaluator.evaluate("What does NovaTech develop?", [self.chunks[0]])
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 1.0)
        self.assertEqual(len(res.missing_information), 0)

    def test_insufficient_evaluation_on_unanswerable_query(self) -> None:
        res = self.evaluator.evaluate("Who is the Chief Financial Officer of NovaTech?", [self.chunks[0]])
        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.0)
        self.assertIn("Chief Financial Officer", res.missing_information[0])

    def test_adaptive_recovery_executes_single_focused_pass(self) -> None:
        # Initial chunk only has company overview, lacks founding year
        initial_chunks = [self.chunks[0]]
        full_chunks = self.chunks

        def mock_retriever(query: str, top_k: int) -> list[Chunk]:
            return [c for c in full_chunks if "2019" in c.text or "found" in c.text]

        recovered_chunks, post_res, trace = self.recovery.execute_recovery(
            question="Who founded NovaTech and when was it founded?",
            initial_chunks=initial_chunks,
            missing_information=["founding year"],
            retriever_func=mock_retriever,
            judge_func=lambda q, ch: self.evaluator.evaluate(q, ch),
        )
        self.assertTrue(trace.triggered)
        self.assertGreater(len(recovered_chunks), len(initial_chunks))
        self.assertTrue(trace.post_recovery_sufficiency)


class TestEvidenceFirstGenerationVerifierAndPipeline(unittest.TestCase):
    """Test Grounded Generation, Citation Verifier, and End-to-End Pipeline."""

    def setUp(self) -> None:
        self.chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech develops cloud-based analytics software. NovaTech was founded by Priya Mehta.",
                token_count=14,
                char_count=87,
            ),
        ]
        self.generator = GroundedGenerator()
        self.verifier = CitationVerifier()
        self.pipeline = EvidenceFirstPipeline(self.chunks)

    def test_grounded_generation_produces_claims_with_citations(self) -> None:
        rec = self.generator.generate("What does NovaTech develop?", self.chunks, is_sufficient=True)
        self.assertFalse(rec.is_abstained)
        self.assertIn("analytics software", rec.answer.lower())
        self.assertGreater(len(rec.claims), 0)
        self.assertEqual(rec.claims[0].source_chunk_ids, ("c1",))

    def test_citation_verifier_validates_entailed_claim(self) -> None:
        rec = self.generator.generate("What does NovaTech develop?", self.chunks, is_sufficient=True)
        report = self.verifier.verify_claims(rec, self.chunks)
        self.assertTrue(report.all_supported)
        self.assertEqual(report.overall_support_score, 1.0)

    def test_pipeline_answers_sufficient_question(self) -> None:
        trace = self.pipeline.run("What does NovaTech develop?")
        self.assertEqual(trace.final_decision, "ANSWERED")
        self.assertIn("analytics software", trace.final_answer.lower())
        self.assertGreater(trace.latency_ms, 0.0)

    def test_pipeline_abstains_on_unanswerable_question(self) -> None:
        trace = self.pipeline.run("Who is the CEO of NovaTech?")
        self.assertEqual(trace.final_decision, "ABSTAINED")
        self.assertIn("cannot be established", trace.final_answer.lower())


class TestEvidenceFirstEvaluationAndBenchmark(unittest.TestCase):
    """Test Person 3 Selective Answering, Baselines, and Benchmark dataset."""

    def test_selective_answering_policy_curve(self) -> None:
        policy = SelectiveAnsweringPolicy()
        records = [
            (1.0, 0.95, True),
            (1.0, 0.90, True),
            (0.0, 0.0, False),
            (0.0, 0.0, False),
        ]
        report = policy.evaluate_curve(records)
        self.assertGreater(report.auacc, 0.5)
        self.assertGreaterEqual(report.coverage_at_90_acc, 0.5)

    def test_baseline_evaluator_recovery_gain(self) -> None:
        chunks = [
            Chunk("c1", "company", "company.txt", 1, "Overview", "NovaTech is a software company founded by Priya Mehta.", 10, 50),
            Chunk("c2", "history", "history.txt", 1, "History", "NovaTech was founded in 2019 in Bengaluru.", 8, 40),
            Chunk("c3", "finance", "finance.txt", 1, "Finance", "NovaTech reported annual revenue of 42 million dollars in 2025.", 10, 60),
        ]
        evaluator = BaselineEvaluator(chunks)
        b0, b1, e1 = evaluator.run_suite()
        self.assertGreaterEqual(e1.accuracy, b1.accuracy)
        self.assertGreaterEqual(b1.accuracy, b0.accuracy)
        self.assertIsNotNone(e1.recovery_gain)
        self.assertGreaterEqual(e1.recovery_gain, 0)

    def test_expanded_benchmark_integrity(self) -> None:
        csv_path = ROOT / "data" / "eval" / "annotated_questions.csv"
        self.assertTrue(csv_path.exists(), "annotated_questions.csv must exist")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 105, "Expanded benchmark must contain exactly 105 items")

        # Check balance
        suff_count = sum(1 for r in reader if r["candidate_label"] == "SUFFICIENT")
        insuff_count = sum(1 for r in reader if r["candidate_label"] == "INSUFFICIENT")
        self.assertTrue(45 <= suff_count <= 60, f"Expected balanced labels, got {suff_count} sufficient")
        self.assertTrue(45 <= insuff_count <= 60, f"Expected balanced labels, got {insuff_count} insufficient")

        # Check all gold labels are strictly PENDING
        for r in reader:
            self.assertEqual(r["gold_label"], "PENDING", f"Gold label must be PENDING for {r['question_id']}")
            self.assertEqual(r["annotation_status"], "PENDING HUMAN ACTION")

        # Check double-annotation subset
        double_annotated = [r for r in reader if r["double_annotated"] == "True"]
        self.assertTrue(20 <= len(double_annotated) <= 35, f"Expected 20-30% double-annotation, got {len(double_annotated)}")


if __name__ == "__main__":
    unittest.main()
