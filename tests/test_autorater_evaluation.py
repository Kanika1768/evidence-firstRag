"""Unit tests for Phase 1 & 2: LLM AutoRater Evaluation."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.local_llm_client import LocalTransformersLLMClient
from agentic_rag.adapters.structured_sufficiency import (
    LLMSufficiencyAutoRater,
    StructuredSufficiencyResult,
)
from agentic_rag.contracts import ContextAssessment, ContextStatus
from scripts.evaluate_autorater import compute_binary_metrics


class TestAutoRaterEvaluation(unittest.TestCase):
    """Test suite for LLM AutoRater evaluation utilities and offline client."""

    def test_compute_binary_metrics(self) -> None:
        y_true = ["SUFFICIENT", "SUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"]
        y_pred = ["SUFFICIENT", "INSUFFICIENT", "INSUFFICIENT", "SUFFICIENT"]

        metrics = compute_binary_metrics(y_true, y_pred)
        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_enum_check_prevents_insufficient_substring_false_positive(self) -> None:
        """Regression test: 'ContextStatus.INSUFFICIENT' must not evaluate to SUFFICIENT via substring match."""
        assessment = ContextAssessment(
            status=ContextStatus.INSUFFICIENT,
            sufficiency_score=0.0,
            missing_facts=("missing fact",),
            reason="Lacks evidence.",
        )
        # Buggy check was: "sufficient" in str(assessment.status).lower() -> True
        # Correct check is:
        status_str = "SUFFICIENT" if assessment.status == ContextStatus.SUFFICIENT else "INSUFFICIENT"
        self.assertEqual(status_str, "INSUFFICIENT")

        suff_assessment = ContextAssessment(
            status=ContextStatus.SUFFICIENT,
            sufficiency_score=1.0,
            missing_facts=(),
            reason="Has evidence.",
        )
        suff_status = "SUFFICIENT" if suff_assessment.status == ContextStatus.SUFFICIENT else "INSUFFICIENT"
        self.assertEqual(suff_status, "SUFFICIENT")

    def test_single_class_slice_metrics(self) -> None:
        """Verify behavior on negative-only slice (e.g. missing intermediate facts)."""
        y_true_neg = ["INSUFFICIENT"] * 10
        y_pred_correct = ["INSUFFICIENT"] * 8 + ["SUFFICIENT"] * 2

        m = compute_binary_metrics(y_true_neg, y_pred_correct)
        self.assertEqual(m["total"], 10)
        self.assertEqual(m["tn"], 8)
        self.assertEqual(m["fp"], 2)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["accuracy"], 0.8)
        # In negative-only slice, TNR (specificity) = 8/10 = 0.8
        tnr = m["tn"] / (m["tn"] + m["fp"])
        self.assertEqual(tnr, 0.8)

    def test_local_transformers_client_caching(self) -> None:
        """Verify client retrieves valid cached model completions with provenance without requiring transformers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LocalTransformersLLMClient(cache_dir=tmpdir, use_cache=True)
            prompt = "Is this context sufficient for test?"
            expected_response = (
                '{"sufficient": true, "probability": 1.0, "missing_information": [], '
                '"rationale": "Direct evidence present."}'
            )

            # 1. Pre-seed a valid model completion into cache
            key = client._cache_key(prompt)
            client._save_to_cache(key, prompt, expected_response, provenance="qwen_local_inference")

            # Check cache file exists on disk with correct schema
            cache_file = Path(tmpdir) / f"{key}.json"
            self.assertTrue(cache_file.exists())
            with open(cache_file, encoding="utf-8") as f:
                cached_data = json.load(f)
            self.assertEqual(cached_data["response"], expected_response)
            self.assertEqual(cached_data["model_name"], client.model_name)
            self.assertEqual(cached_data["provenance"], "qwen_local_inference")
            self.assertIn("timestamp", cached_data)

            # 2. Verify that cached Qwen response returns correct response and provenance
            resp, prov = client.generate_with_provenance(prompt)
            self.assertEqual(resp, expected_response)
            self.assertEqual(prov, "qwen_local_inference_cache_hit")
            self.assertEqual(client.last_provenance, "qwen_local_inference_cache_hit")

            # 3. Verify __call__ returns cached text directly
            self.assertEqual(client(prompt), expected_response)

    def test_local_transformers_client_mock_inference_and_cache_cycle(self) -> None:
        """Verify that model generation writes to cache and subsequent call hits cache."""
        fake_torch = types.ModuleType("torch")

        class _FakeNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        fake_torch.no_grad = _FakeNoGrad

        with tempfile.TemporaryDirectory() as tmpdir:
            client = LocalTransformersLLMClient(cache_dir=tmpdir, use_cache=True)
            prompt = "Is this context sufficient?"
            mock_model_output = '{"sufficient": true, "probability": 0.9, "missing_information": [], "rationale": "ok"}'

            # Mock tokenizer and model so it does not import transformers or load weights
            client._ensure_model_loaded = lambda: True
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template.return_value = "templated prompt"
            mock_inputs = MagicMock()
            mock_inputs.input_ids = MagicMock(shape=[1, 5])
            mock_inputs.keys.return_value = ["input_ids"]
            mock_inputs.__getitem__.return_value = mock_inputs.input_ids
            mock_tokenizer.return_value = mock_inputs
            mock_tokenizer.decode.return_value = mock_model_output
            client._tokenizer = mock_tokenizer

            mock_model = MagicMock()
            mock_model.generate.return_value = [[0] * 10]
            client._model = mock_model

            # First call: executes mock model inference within patched torch environment, writes to cache
            with patch.dict(sys.modules, {"torch": fake_torch}):
                resp1, prov1 = client.generate_with_provenance(prompt)
            self.assertEqual(resp1, mock_model_output)
            self.assertEqual(prov1, "qwen_local_inference")

            key = client._cache_key(prompt)
            cache_file = Path(tmpdir) / f"{key}.json"
            self.assertTrue(cache_file.exists())

            # Detach model so inference would fail if cache were missed
            client._model = None
            client._ensure_model_loaded = lambda: False

            # Second call: hits cache, returning provenance tag qwen_local_inference_cache_hit
            resp2, prov2 = client.generate_with_provenance(prompt)
            self.assertEqual(resp2, mock_model_output)
            self.assertEqual(prov2, "qwen_local_inference_cache_hit")

    def test_autorater_with_mock_client(self) -> None:
        def mock_client(prompt: str) -> str:
            return (
                '{"sufficient": true, "probability": 0.95, '
                '"missing_information": [], "rationale": "Clear evidence"}'
            )

        autorater = LLMSufficiencyAutoRater(llm_client=mock_client)
        result = autorater.evaluate(
            question="Where is Paris?",
            evidence="Paris is the capital of France.",
        )
        self.assertIsInstance(result, StructuredSufficiencyResult)
        self.assertTrue(result.sufficient)
        self.assertEqual(result.probability, 0.95)
        self.assertEqual(result.missing_information, [])

    def test_provenance_tracking_and_no_fallback_caching(self) -> None:
        """Verify fallback responses are never saved to cache and are tagged with provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            client = LocalTransformersLLMClient(
                model_name="nonexistent/fake-model",
                cache_dir=tmpdir,
                use_cache=True,
                allow_fallback=True,
            )
            # Ensure model loading fails immediately without attempting remote connections
            client._ensure_model_loaded = lambda: False

            prompt = "What is George Rankin's occupation?"
            resp, prov = client.generate_with_provenance(prompt)
            self.assertEqual(prov, "heuristic_fallback")
            self.assertIn("sufficient", resp)

            # CRITICAL: Verify NO cache file was written for the fallback response
            key = client._cache_key(prompt)
            cache_file = Path(tmpdir) / f"{key}.json"
            self.assertFalse(cache_file.exists())

    def test_structured_sufficiency_fallback_provenance(self) -> None:
        """Verify LLMSufficiencyAutoRater tags results with fallback mode and provenance."""
        def failing_client(prompt: str) -> str:
            raise RuntimeError("LLM connection timed out")

        autorater = LLMSufficiencyAutoRater(
            llm_client=failing_client,
            fallback_judge=lambda q, ev: StructuredSufficiencyResult(
                sufficient=False,
                probability=0.0,
                missing_information=["fallback missing fact"],
                rationale="Fallback decision.",
            ),
        )
        res = autorater.evaluate(question="Where is Paris?", evidence="Paris is in France.")
        self.assertIn("Fallback", res.evaluator_mode)
        self.assertTrue(res.provenance.startswith("heuristic_fallback"))
        d = res.to_dict()
        self.assertIn("provenance", d)
        self.assertEqual(d["provenance"], res.provenance)
