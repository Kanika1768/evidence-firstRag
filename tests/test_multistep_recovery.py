"""Unit tests for Phase 4: Multi-Step Cross-Document Recovery."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.structured_sufficiency import StructuredSufficiencyResult
from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.recovery import AdaptiveRecoveryController, HopTrace, RecoveryTrace


class TestMultiStepRecovery(unittest.TestCase):
    """Test suite for multi-step cross-document evidence recovery."""

    def setUp(self) -> None:
        self.doc_a = Chunk(
            chunk_id="doc_a:c1",
            document_id="doc_a",
            document_name="doc_a.txt",
            page=1,
            section="Leadership",
            text="Project Orion was founded by Dr. Evelyn Reed.",
            token_count=8,
            char_count=45,
        )
        self.doc_b = Chunk(
            chunk_id="doc_b:c1",
            document_id="doc_b",
            document_name="doc_b.txt",
            page=1,
            section="Awards",
            text="Dr. Evelyn Reed received the Turing Medal in 2022.",
            token_count=9,
            char_count=50,
        )
        self.doc_c = Chunk(
            chunk_id="doc_c:c1",
            document_id="doc_c",
            document_name="doc_c.txt",
            page=1,
            section="Ceremony",
            text="The Turing Medal in 2022 was presented in Zurich, Switzerland.",
            token_count=10,
            char_count=62,
        )

    def test_multi_hop_cross_document_recovery(self) -> None:
        """Verify that multi-step recovery chains evidence across 3 documents."""
        all_chunks = [self.doc_a, self.doc_b, self.doc_c]
        question = "In which city was the award presented to the founder of Project Orion?"

        # Mock retriever indexing the 3 documents
        def mock_retriever(query: str, top_k: int) -> list[Chunk]:
            q_lower = query.lower()
            matched = []
            for c in all_chunks:
                # Match keywords
                words = [w for w in q_lower.split() if len(w) > 3]
                if any(w in c.text.lower() for w in words):
                    matched.append(c)
            return matched[:top_k]

        # Mock judge checking for founder, award, and city
        def mock_judge(q: str, evidence: Sequence[Chunk]) -> StructuredSufficiencyResult:
            texts = " ".join(c.text for c in evidence)
            has_founder = "Evelyn Reed" in texts
            has_award = "Turing Medal" in texts
            has_city = "Zurich" in texts

            missing = []
            if not has_founder:
                missing.append("founder of Project Orion")
            if not has_award:
                missing.append("award received by founder Evelyn Reed")
            if not has_city:
                missing.append("presentation city of Turing Medal 2022")

            sufficient = has_founder and has_award and has_city
            return StructuredSufficiencyResult(
                sufficient=sufficient,
                probability=1.0 if sufficient else 0.3,
                missing_information=missing,
                rationale="Complete chain found." if sufficient else "Missing facts in chain.",
            )

        # Baseline: Depth K=1 only finds doc_b (still missing city in doc_c)
        controller_k1 = AdaptiveRecoveryController(max_depth=1)
        res_chunks_1, eval_1, trace_1 = controller_k1.execute_recovery(
            question=question,
            initial_chunks=[self.doc_a],
            missing_information=["award received by founder Evelyn Reed", "presentation city"],
            retriever_func=mock_retriever,
            judge_func=mock_judge,
        )
        self.assertFalse(trace_1.post_recovery_sufficiency)
        self.assertEqual(trace_1.total_hops_executed, 1)

        # Multi-step: Depth K=2 chains doc_a -> doc_b -> doc_c successfully
        controller_k2 = AdaptiveRecoveryController(max_depth=2)
        res_chunks_2, eval_2, trace_2 = controller_k2.execute_recovery(
            question=question,
            initial_chunks=[self.doc_a],
            missing_information=["award received by founder Evelyn Reed", "presentation city"],
            retriever_func=mock_retriever,
            judge_func=mock_judge,
        )
        self.assertTrue(trace_2.post_recovery_sufficiency)
        self.assertEqual(trace_2.total_hops_executed, 2)
        self.assertEqual(trace_2.termination_reason, "SUFFICIENT_CONVERGENCE")
        self.assertEqual(len(trace_2.hop_traces), 2)
        self.assertTrue(any("Zurich" in c.text for c in res_chunks_2))

    def test_cycle_detection(self) -> None:
        """Verify that identical query reformulations trigger cycle detection."""
        controller = AdaptiveRecoveryController(max_depth=3)

        def mock_static_retriever(query: str, top_k: int) -> list[Chunk]:
            return [self.doc_b]

        def mock_always_missing_judge(q: str, evidence: Sequence[Chunk]) -> StructuredSufficiencyResult:
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.2,
                missing_information=["static missing fact"],
                rationale="Still missing.",
            )

        _, _, trace = controller.execute_recovery(
            question="What is the mission?",
            initial_chunks=[self.doc_a],
            missing_information=["static missing fact"],
            retriever_func=mock_static_retriever,
            judge_func=mock_always_missing_judge,
        )

        # Hop 1 executed, Hop 2 repeats query -> cycle detected before executing retrieval
        self.assertTrue(trace.cycle_detected)
        self.assertEqual(trace.termination_reason, "CYCLE_DETECTED")
        self.assertEqual(trace.total_hops_executed, 1)
        self.assertEqual(trace.attempts, 1)

    def test_negative_max_depth_rejection(self) -> None:
        """Verify that negative max_depth is rejected at initialization and per-call."""
        with self.assertRaises(ValueError):
            AdaptiveRecoveryController(max_depth=-1)

        controller = AdaptiveRecoveryController(max_depth=2)
        with self.assertRaises(ValueError):
            controller.execute_recovery(
                question="What is the mission?",
                initial_chunks=[self.doc_a],
                missing_information=["static missing fact"],
                retriever_func=lambda q, top_k: [self.doc_b],
                judge_func=lambda q, ch: StructuredSufficiencyResult(False, 0.0, ["static missing fact"]),
                max_depth=-1,
            )

    def test_duplicate_chunk_handling(self) -> None:
        """Verify that duplicate chunks are pruned across hops."""
        controller = AdaptiveRecoveryController(max_depth=2)

        def mock_duplicate_retriever(query: str, top_k: int) -> list[Chunk]:
            return [self.doc_a]  # Returns same chunk as initial

        def mock_judge(q: str, evidence: Sequence[Chunk]) -> StructuredSufficiencyResult:
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.1,
                missing_information=["unfound fact"],
            )

        res_chunks, _, trace = controller.execute_recovery(
            question="Tell me about Orion",
            initial_chunks=[self.doc_a],
            missing_information=["unfound fact"],
            retriever_func=mock_duplicate_retriever,
            judge_func=mock_judge,
        )

        self.assertEqual(len(res_chunks), 1)
        self.assertEqual(trace.termination_reason, "NO_NEW_CHUNKS")

    def test_downstream_entity_unavailable_until_intermediate_discovery(self) -> None:
        """Verify downstream evidence is inaccessible until intermediate bridge is discovered."""
        doc_a = Chunk(
            chunk_id="doc_a:c1",
            document_id="doc_a",
            document_name="doc_a.txt",
            page=1,
            section="Overview",
            text="TechNova operates an advanced cyber-defense laboratory named CybroShield Labs.",
            token_count=10,
            char_count=80,
        )
        doc_b = Chunk(
            chunk_id="doc_b:c1",
            document_id="doc_b",
            document_name="doc_b.txt",
            page=1,
            section="Leadership",
            text="CybroShield Labs is headed by Principal Scientist Dr. Aris Thorne.",
            token_count=10,
            char_count=74,
        )
        doc_c = Chunk(
            chunk_id="doc_c:c1",
            document_id="doc_c",
            document_name="doc_c.txt",
            page=1,
            section="Patents",
            text="Dr. Aris Thorne registered the patent for Autonomous Zero-Trust Verifier in Munich.",
            token_count=12,
            char_count=87,
        )

        corpus = [doc_a, doc_b, doc_c]
        question = "In which city was the patent registered by the head of TechNova's cyber-defense lab?"

        def retriever(query: str, top_k: int = 6) -> list[Chunk]:
            import re
            q_tokens = set(re.findall(r"[a-z0-9\-]+", query.lower()))
            scored = []
            for c in corpus:
                c_tokens = set(re.findall(r"[a-z0-9\-]+", c.text.lower()))
                overlap = len(q_tokens & c_tokens)
                if overlap > 0:
                    scored.append((c, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, _ in scored[:top_k]]

        def stepwise_judge(q: str, evidence: Sequence[Chunk]) -> StructuredSufficiencyResult:
            ev_text = " ".join(c.text.lower() for c in evidence)
            has_head = "aris thorne" in ev_text
            has_patent_city = "munich" in ev_text

            if not has_head:
                return StructuredSufficiencyResult(
                    sufficient=False,
                    probability=0.3,
                    missing_information=["head of CybroShield Labs"],
                    rationale="Missing lab leadership.",
                )
            if not has_patent_city:
                return StructuredSufficiencyResult(
                    sufficient=False,
                    probability=0.6,
                    missing_information=["patent registration city for Dr. Aris Thorne"],
                    rationale="Missing patent city.",
                )
            return StructuredSufficiencyResult(
                sufficient=True,
                probability=1.0,
                missing_information=[],
                rationale="Complete evidence chain established.",
            )

        # 1. Depth K=1 only discovers Doc B (bridge entity)
        controller_k1 = AdaptiveRecoveryController(max_depth=1)
        res_k1, eval_k1, trace_k1 = controller_k1.execute_recovery(
            question=question,
            initial_chunks=[doc_a],
            missing_information=["head of CybroShield Labs"],
            retriever_func=retriever,
            judge_func=stepwise_judge,
        )
        self.assertFalse(trace_k1.post_recovery_sufficiency)
        self.assertEqual(trace_k1.total_hops_executed, 1)
        self.assertEqual([c.chunk_id for c in trace_k1.recovered_chunks], ["doc_b:c1"])
        # Crucial: Doc C is NOT present after hop 1
        self.assertNotIn("doc_c:c1", [c.chunk_id for c in res_k1])

        # 2. Depth K=2 uses discovered bridge entity from Doc B to reach Doc C
        controller_k2 = AdaptiveRecoveryController(max_depth=2)
        res_k2, eval_k2, trace_k2 = controller_k2.execute_recovery(
            question=question,
            initial_chunks=[doc_a],
            missing_information=["head of CybroShield Labs"],
            retriever_func=retriever,
            judge_func=stepwise_judge,
        )
        self.assertTrue(trace_k2.post_recovery_sufficiency)
        self.assertEqual(trace_k2.total_hops_executed, 2)
        self.assertEqual(trace_k2.termination_reason, "SUFFICIENT_CONVERGENCE")
        self.assertIn("doc_c:c1", [c.chunk_id for c in res_k2])
        # Hop 1 query targeted the lab head; Hop 2 query used bridge terms / missing fact (patent, city)
        self.assertTrue(any(term in trace_k2.hop_traces[1].formulated_query.lower() for term in ["patent", "city", "dr"]))
