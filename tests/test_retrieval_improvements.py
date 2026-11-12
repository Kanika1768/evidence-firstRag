"""Unit tests for Phase 3: Difficult-Query Retrieval Improvements."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.retrieval import (
    EXTENDED_RETRIEVAL_MODES,
    CrossEncoderReranker,
    DenseEmbeddingModel,
    HybridRetriever,
    NISTTerminologyExpander,
    _load_cross_encoder,
    _load_sentence_transformer,
    adaptive_score_fusion,
)


class TestRetrievalImprovements(unittest.TestCase):
    """Test suite for adaptive score fusion, terminology expander, and extended modes."""

    def test_terminology_expansion(self) -> None:
        q1 = "What does AI RMF 1.0 say about risk?"
        expanded1 = NISTTerminologyExpander.expand_query(q1)
        self.assertIn("artificial intelligence risk management framework", expanded1.lower())

        q2 = "Explain the TEVV process for generative models."
        expanded2 = NISTTerminologyExpander.expand_query(q2)
        self.assertIn("test evaluation verification and validation", expanded2.lower())

        # Clean query without mapped acronyms remains untouched
        q3 = "What is the capital of France?"
        self.assertEqual(NISTTerminologyExpander.expand_query(q3), q3)

    def test_adaptive_score_fusion_logic(self) -> None:
        dense_hits = [("c1", 0.9), ("c2", 0.7), ("c3", 0.4)]
        bm25_hits = [("c2", 15.0), ("c4", 12.0), ("c1", 8.0)]

        fused = adaptive_score_fusion(dense_hits, bm25_hits, alpha=0.5, cooccurrence_bonus=0.2)
        self.assertTrue(len(fused) == 4)

        # Chunks present in both dense and BM25 receive cooccurrence bonus
        fused_cids = [cid for cid, _ in fused]
        self.assertIn("c1", fused_cids)
        self.assertIn("c2", fused_cids)
        # c2 is high in both, so it should be near or at top
        self.assertEqual(fused_cids[0], "c2")

    def test_adaptive_score_fusion_deterministic_tie_ordering(self) -> None:
        """Verify that tied fusion scores are ordered deterministically by chunk_id ascending."""
        # Both c_beta and c_alpha will have identical dense and bm25 scores
        dense_hits = [("c_beta", 0.5), ("c_alpha", 0.5), ("c_gamma", 0.2)]
        bm25_hits = [("c_beta", 10.0), ("c_alpha", 10.0), ("c_gamma", 4.0)]

        fused = adaptive_score_fusion(dense_hits, bm25_hits, alpha=0.5, cooccurrence_bonus=0.1)
        # c_beta and c_alpha tie on score; chunk_id ascending requires c_alpha before c_beta
        self.assertEqual(fused[0][0], "c_alpha")
        self.assertEqual(fused[1][0], "c_beta")
        self.assertAlmostEqual(fused[0][1], fused[1][1], places=6)
        self.assertEqual(fused[2][0], "c_gamma")

    def test_extended_modes_retrieval(self) -> None:
        chunks = [
            Chunk(
                chunk_id="c1",
                document_id="d1",
                document_name="nist.ai.100-1.pdf",
                page=1,
                section="AI RMF Core",
                text="Artificial Intelligence Risk Management Framework provides guidance.",
                token_count=10,
                char_count=70,
            ),
            Chunk(
                chunk_id="c2",
                document_id="d1",
                document_name="nist.ai.100-1.pdf",
                page=2,
                section="TEVV",
                text="TEVV encompasses testing, evaluation, verification, and validation.",
                token_count=8,
                char_count=68,
            ),
        ]
        retriever = HybridRetriever(chunks)

        for mode in ("adaptive_hybrid", "adaptive_hybrid_rerank"):
            self.assertIn(mode, EXTENDED_RETRIEVAL_MODES)
            results = retriever.retrieve("What is TEVV?", top_k=2, mode=mode)
            self.assertTrue(len(results) > 0)
            self.assertEqual(results[0].chunk_id, "c2")

    def test_unavailable_local_model_offline_fallback(self) -> None:
        """Verify an unavailable local model returns None and activates deterministic fallback without network."""
        missing_model = "test-mock-org/completely-nonexistent-embed-model-offline"

        # Loader returns None gracefully without throwing or attempting online download
        self.assertIsNone(_load_sentence_transformer(missing_model))

        # DenseEmbeddingModel falls back cleanly to feature-hash-fallback
        dense_model = DenseEmbeddingModel(missing_model)
        self.assertEqual(dense_model.backend_name, "feature-hash-fallback")
        vec = dense_model.encode_query("AI Risk Management")
        self.assertEqual(len(vec), 384)
        self.assertTrue(all(isinstance(v, float) for v in vec))

        # CrossEncoder loader and fallback
        missing_reranker = "test-mock-org/completely-nonexistent-reranker-offline"
        self.assertIsNone(_load_cross_encoder(missing_reranker))

        reranker = CrossEncoderReranker(missing_reranker)
        self.assertEqual(reranker.backend_name, "lexical-token-overlap-fallback")

        sample_chunk = Chunk(
            chunk_id="test-c1",
            document_id="test-doc",
            document_name="test.txt",
            page=1,
            section="TEVV",
            text="TEVV is testing evaluation verification validation.",
            token_count=7,
            char_count=52,
        )
        ranked = reranker.rerank("TEVV", [sample_chunk], top_k=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0][0].chunk_id, "test-c1")

    def test_sentence_transformer_loader_never_retries_online_on_local_failure(self) -> None:
        """Verify that when local loading fails, SentenceTransformer is NEVER called without local_files_only=True."""
        model_name = "test-org/fake-local-files-only-enforced-model"
        _load_sentence_transformer.cache_clear()

        mock_st = MagicMock(side_effect=OSError("Cannot find files in local cache"))
        fake_st_mod = types.ModuleType("sentence_transformers")
        fake_st_mod.SentenceTransformer = mock_st

        with patch.dict(sys.modules, {"sentence_transformers": fake_st_mod}):
            _load_sentence_transformer.cache_clear()
            try:
                result = _load_sentence_transformer(model_name)
                self.assertIsNone(result)

                # Proves SentenceTransformer was called strictly once with local_files_only=True
                # and was NEVER retried online without local_files_only=True
                self.assertEqual(mock_st.call_count, 1)
                args, kwargs = mock_st.call_args
                self.assertEqual(args, (model_name,))
                self.assertTrue(kwargs.get("local_files_only", False))
            finally:
                _load_sentence_transformer.cache_clear()

    def test_cross_encoder_loader_never_retries_online_on_local_failure(self) -> None:
        """Verify that when local loading fails, CrossEncoder is NEVER called without local_files_only=True."""
        model_name = "test-org/fake-local-files-only-enforced-reranker"
        _load_cross_encoder.cache_clear()

        mock_ce = MagicMock(side_effect=OSError("Cannot find files in local cache"))
        fake_ce_mod = types.ModuleType("sentence_transformers")
        fake_ce_mod.CrossEncoder = mock_ce

        with patch.dict(sys.modules, {"sentence_transformers": fake_ce_mod}):
            _load_cross_encoder.cache_clear()
            try:
                result = _load_cross_encoder(model_name)
                self.assertIsNone(result)

                # Proves CrossEncoder was called strictly once with local_files_only=True
                # and was NEVER retried online without local_files_only=True
                self.assertEqual(mock_ce.call_count, 1)
                args, kwargs = mock_ce.call_args
                self.assertEqual(args, (model_name,))
                self.assertEqual(kwargs.get("max_length"), 512)
                self.assertTrue(kwargs.get("local_files_only", False))
            finally:
                _load_cross_encoder.cache_clear()
