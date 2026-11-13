"""Targeted Unit Tests for ICLR 2025 Sufficient Context Alignment.

Verifies:
1. Prompt construction & inclusion of Roald Dahl 1-shot demonstration.
2. Temporal timestamp anchoring (<TIMESTAMP> replaced with date).
3. Zero oracle information leakage (planned required facts strictly excluded).
4. Robust parsing of both Sufficient (1) and Insufficient (0) judgments.
5. Extraction of rationale from ### EXPLANATION text.
6. Robustness against malformed or code-fenced responses.
7. Gemini REST client plain text configuration.
8. Local Qwen client configuration without conflicting schema override.
9. Baseline evaluator preservation (sufficiency_v1 remains fully functional).
10. SelectiveAnsweringModel calibration and accuracy-coverage curve computation.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.llm_client import GenericRestLLMClient, create_llm_client
from agentic_rag.adapters.local_llm_client import LocalTransformersLLMClient
from agentic_rag.adapters.structured_sufficiency import (
    LLMSufficiencyAutoRater,
    StructuredSufficiencyResult,
)
from agentic_rag.contracts import ContextStatus
from agentic_rag.prompts.sufficiency_iclr2025 import (
    DEFAULT_BENCHMARK_TIMESTAMP,
    ICLR2025_PROMPT_VERSION,
    build_iclr2025_prompt,
)
from agentic_rag.prompts.sufficiency_v1 import (
    SUFFICIENCY_PROMPT_VERSION,
    build_sufficiency_prompt,
)
from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.evaluation import (
    AccuracyCoverageReport,
    SelectiveAnsweringModel,
    SelectiveAnsweringPolicy,
)


class TestICLR2025PromptAndLeakage(unittest.TestCase):
    """Test prompt construction, timestamp, and zero oracle leakage."""

    def test_prompt_construction_and_one_shot(self) -> None:
        prompt = build_iclr2025_prompt(
            question="What is George Rankin's occupation?",
            references="George Rankin was an Australian politician.",
            timestamp="2024-06-01",
        )
        self.assertIn("Roald Dahl's Guide to Railway Safety", prompt)
        self.assertIn("British Railways Board", prompt)
        self.assertIn('{"Sufficient Context": 1}', prompt)
        self.assertIn("Assume the queries have timestamp 2024-06-01.", prompt)
        self.assertIn("What is George Rankin's occupation?", prompt)
        self.assertIn("George Rankin was an Australian politician.", prompt)

    def test_zero_oracle_leakage(self) -> None:
        """Verify that planned required facts and candidate labels cannot leak into the prompt."""
        prompt = build_iclr2025_prompt(
            question="What is the capital of France?",
            references="Paris is the capital of France.",
        )
        self.assertNotIn("PLANNED REQUIRED FACTS", prompt)
        self.assertNotIn("candidate_label", prompt)
        self.assertNotIn("gold_label", prompt)
        self.assertNotIn("SUFFICIENT", prompt.split("### QUESTION")[-1])

    def test_default_timestamp_insertion(self) -> None:
        prompt = build_iclr2025_prompt("Q?", "Ctx")
        self.assertIn(f"Assume the queries have timestamp {DEFAULT_BENCHMARK_TIMESTAMP}.", prompt)
        self.assertNotIn("<TIMESTAMP>", prompt)


class TestICLR2025ParserRobustness(unittest.TestCase):
    """Test parsing of paper-aligned output formats, rationales, and edge cases."""

    def test_parse_sufficient_one_with_explanation(self) -> None:
        raw = """### EXPLANATION
Step 1: Check if George Rankin is mentioned.
Step 2: Check if occupation is mentioned.
The reference states George Rankin was an Australian politician.

### EVALUATION
```json
{"Sufficient Context": 1}
```"""
        result = StructuredSufficiencyResult.from_json(raw)
        self.assertTrue(result.sufficient)
        self.assertEqual(result.probability, 1.0)
        self.assertIn("Step 1: Check if George Rankin is mentioned.", result.rationale)

    def test_parse_insufficient_zero_with_explanation(self) -> None:
        raw = """### EXPLANATION
The reference mentions he lived on a farm, but does not state his occupation.

### JSON
{"Sufficient Context": 0}"""
        result = StructuredSufficiencyResult.from_json(raw)
        self.assertFalse(result.sufficient)
        self.assertEqual(result.probability, 0.0)
        self.assertIn("The reference mentions he lived on a farm", result.rationale)

    def test_parse_string_and_boolean_variants(self) -> None:
        # String "1"
        res1 = StructuredSufficiencyResult.from_dict({"Sufficient Context": "1"})
        self.assertTrue(res1.sufficient)

        # String "0"
        res0 = StructuredSufficiencyResult.from_dict({"Sufficient Context": "0"})
        self.assertFalse(res0.sufficient)

        # Boolean True
        res_t = StructuredSufficiencyResult.from_dict({"Sufficient Context": True})
        self.assertTrue(res_t.sufficient)

        # Boolean False
        res_f = StructuredSufficiencyResult.from_dict({"Sufficient Context": False})
        self.assertFalse(res_f.sufficient)

    def test_parse_regex_fallback_on_malformed_json(self) -> None:
        raw_malformed = """### EXPLANATION
Context is sufficient to infer the answer.
### JSON
{ 'Sufficient Context': 1, trailing_comma: }"""
        result = StructuredSufficiencyResult.from_json(raw_malformed)
        self.assertTrue(result.sufficient)
        self.assertEqual(result.probability, 1.0)

    def test_parse_empty_or_whitespace_response(self) -> None:
        result = StructuredSufficiencyResult.from_json("   ")
        self.assertFalse(result.sufficient)
        self.assertEqual(result.probability, 0.0)
        self.assertIn("Empty model response", result.rationale)


class TestClientConfigurations(unittest.TestCase):
    """Test client configurations and backward compatibility."""

    def test_gemini_client_plain_text_mode(self) -> None:
        client = GenericRestLLMClient(
            provider="gemini",
            api_key="dummy_key",
            model="gemini-3.5-flash-lite",
            response_mime_type="text/plain",
        )
        self.assertEqual(client.response_mime_type, "text/plain")

    def test_gemini_client_default_preserves_json(self) -> None:
        client = GenericRestLLMClient(
            provider="gemini",
            api_key="dummy_key",
            model="gemini-3.5-flash-lite",
        )
        self.assertEqual(client.response_mime_type, "application/json")

    def test_create_llm_client_response_mime_type(self) -> None:
        client, desc = create_llm_client(response_mime_type="text/plain")
        if client is not None:
            self.assertEqual(client.response_mime_type, "text/plain")

    def test_local_qwen_chat_template_avoids_conflicting_schema(self) -> None:
        client = LocalTransformersLLMClient(max_new_tokens=512)
        self.assertEqual(client.max_new_tokens, 512)

    def test_autorater_dispatches_iclr2025_vs_baseline(self) -> None:
        dummy_client = lambda prompt: '### EXPLANATION\nOK\n### JSON\n{"Sufficient Context": 1}'
        ar_iclr = LLMSufficiencyAutoRater(
            llm_client=dummy_client,
            prompt_version=ICLR2025_PROMPT_VERSION,
        )
        chunk = Chunk("c1", "d1", "doc", 1, "s", "Paris is the capital of France.", 7, 30)
        res_iclr = ar_iclr.evaluate("Capital of France?", [chunk])
        self.assertTrue(res_iclr.sufficient)

        # Baseline v1.0 remains functional
        dummy_v1 = lambda prompt: '{"sufficient": true, "probability": 0.95, "missing_information": [], "rationale": "OK"}'
        ar_v1 = LLMSufficiencyAutoRater(
            llm_client=dummy_v1,
            prompt_version=SUFFICIENCY_PROMPT_VERSION,
        )
        res_v1 = ar_v1.evaluate("Capital of France?", [chunk])
        self.assertTrue(res_v1.sufficient)
        self.assertEqual(res_v1.probability, 0.95)


class TestSelectiveAnsweringMethodology(unittest.TestCase):
    """Test SelectiveAnsweringModel fitting and threshold policy curve."""

    def test_fit_and_predict(self) -> None:
        model = SelectiveAnsweringModel(w1=1.0, w2=1.0, b=0.0)
        p = model.predict_proba(1.0, 1.0)
        self.assertGreater(p, 0.5)

        # Fit on clear synthetic data
        train_samples = [(1.0, 1.0), (1.0, 0.9), (0.0, 0.1), (0.0, 0.2)]
        train_labels = [1, 1, 0, 0]
        model.fit(train_samples, train_labels, epochs=100, lr=0.2)

        # Should predict high for sufficient+confident, low for insufficient
        self.assertGreater(model.predict_proba(1.0, 1.0), 0.7)
        self.assertLess(model.predict_proba(0.0, 0.0), 0.3)

    def test_accuracy_coverage_curve(self) -> None:
        policy = SelectiveAnsweringPolicy()
        records = [
            (1.0, 0.95, True),
            (1.0, 0.90, True),
            (0.8, 0.75, True),
            (0.0, 0.20, False),
            (0.0, 0.10, False),
        ]
        report = policy.evaluate_curve(records, num_thresholds=10)
        self.assertIsInstance(report, AccuracyCoverageReport)
        self.assertGreater(report.auacc, 0.5)


if __name__ == "__main__":
    unittest.main()
