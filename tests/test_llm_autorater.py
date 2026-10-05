"""Unit tests for pluggable LLM-based Sufficient Context AutoRater.

Verifies:
1. Valid output: structured JSON parsing, probability bounds, missing facts, markdown stripping, dict outputs.
2. Malformed output: trailing commas, single-quote repair, regex field extraction, and unparseable garbage handling.
3. Fallback behavior: exception catching, delegation to deterministic judge, disabled fallback assertions, and pipeline integration.
"""

from __future__ import annotations

import json
import os
import unittest
import urllib.error
from pathlib import Path
from typing import Any

from agentic_rag.adapters.structured_sufficiency import (
    LLMSufficiencyAutoRater,
    StructuredSufficiencyResult,
)
from agentic_rag.contracts import Snippet
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge
from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.pipeline import EvidenceFirstPipeline
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator

ROOT = Path(__file__).resolve().parents[1]


class TestLLMSufficiencyAutoRaterValidOutput(unittest.TestCase):
    """Test suite for valid LLM AutoRater outputs."""

    def setUp(self) -> None:
        self.sample_chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech develops cloud-based analytics software.",
                token_count=8,
                char_count=52,
            )
        ]

    def test_valid_json_output_sufficient(self) -> None:
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "sufficient": True,
                "probability": 0.96,
                "missing_information": [],
                "rationale": "The context directly identifies NovaTech software products.",
            })

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("What does NovaTech develop?", self.sample_chunks)

        self.assertIsInstance(res, StructuredSufficiencyResult)
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 0.96)
        self.assertEqual(res.missing_information, [])
        self.assertIn("NovaTech", res.rationale)

        # Test JSON serialization
        json_str = autorater.evaluate_to_json("What does NovaTech develop?", self.sample_chunks)
        parsed = json.loads(json_str)
        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["probability"], 0.96)
        self.assertEqual(parsed["missing_information"], [])
        self.assertIn("rationale", parsed)

    def test_valid_json_output_insufficient(self) -> None:
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "sufficient": False,
                "probability": 0.25,
                "missing_information": ["Chief Financial Officer name"],
                "rationale": "Context mentions software but no executive staff.",
            })

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Who is the CFO of NovaTech?", self.sample_chunks)

        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.25)
        self.assertEqual(res.missing_information, ["Chief Financial Officer name"])
        self.assertIn("executive staff", res.rationale)

    def test_valid_markdown_wrapped_json(self) -> None:
        def mock_llm(prompt: str) -> str:
            return (
                "```json\n"
                "{\n"
                '  "sufficient": true,\n'
                '  "probability": 1.0,\n'
                '  "missing_information": [],\n'
                '  "rationale": "Found in context."\n'
                "}\n"
                "```"
            )

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("What does NovaTech develop?", self.sample_chunks)
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 1.0)
        self.assertEqual(res.rationale, "Found in context.")

    def test_dict_output_from_llm_client(self) -> None:
        def mock_llm(prompt: str) -> dict[str, Any]:
            return {
                "sufficient": True,
                "probability": 0.90,
                "missing_information": [],
                "rationale": "Directly returns dict.",
            }

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("What does NovaTech develop?", self.sample_chunks)
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 0.90)

    def test_evidence_formats_supported(self) -> None:
        def mock_llm(prompt: str) -> str:
            # Verify evidence text appeared in prompt
            self.assertIn("analytics software", prompt)
            return json.dumps({
                "sufficient": True,
                "probability": 0.95,
                "missing_information": [],
                "rationale": "OK",
            })

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)

        # 1. Chunk objects
        res1 = autorater.evaluate("Query", self.sample_chunks)
        self.assertTrue(res1.sufficient)

        # 2. Snippet objects
        snippets = [Snippet(id="s1", corpus_id="c", document_id="d", text="analytics software")]
        res2 = autorater.evaluate("Query", snippets)
        self.assertTrue(res2.sufficient)

        # 3. List of strings
        res3 = autorater.evaluate("Query", ["analytics software"])
        self.assertTrue(res3.sufficient)

        # 4. Raw string
        res4 = autorater.evaluate("Query", "analytics software")
        self.assertTrue(res4.sufficient)

    def test_empty_evidence_returns_insufficient_without_llm_call(self) -> None:
        called = False

        def mock_llm(prompt: str) -> str:
            nonlocal called
            called = True
            return "{}"

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("What does NovaTech develop?", [])
        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.0)
        self.assertFalse(called)


class TestLLMSufficiencyAutoRaterMalformedOutput(unittest.TestCase):
    """Test suite for malformed and dirty LLM outputs handled via repair parser."""

    def setUp(self) -> None:
        self.sample_chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech develops cloud-based analytics software.",
                token_count=8,
                char_count=52,
            )
        ]

    def test_malformed_json_with_single_quotes_and_trailing_comma(self) -> None:
        def mock_llm(prompt: str) -> str:
            return "{'sufficient': true, 'probability': 0.85, 'missing_information': ['part 1',], 'rationale': 'Repaired',}"

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Question", self.sample_chunks)

        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 0.85)
        self.assertEqual(res.missing_information, ["part 1"])
        self.assertEqual(res.rationale, "Repaired")

    def test_malformed_json_with_surrounding_conversational_text(self) -> None:
        def mock_llm(prompt: str) -> str:
            return (
                "Sure, I can evaluate that for you!\n"
                '{"sufficient": false, "probability": 0.2, "missing_information": ["founding date"], "rationale": "Missing date"}\n'
                "Let me know if you need anything else!"
            )

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Question", self.sample_chunks)

        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.2)
        self.assertEqual(res.missing_information, ["founding date"])

    def test_malformed_regex_field_extraction(self) -> None:
        def mock_llm(prompt: str) -> str:
            return 'sufficient: false, probability: 0.15, rationale: "Lacks core details", missing_information: ["detail A", "detail B"]'

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Question", self.sample_chunks)

        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.15)
        self.assertIn("detail A", res.missing_information)

    def test_completely_unparseable_garbage_handled_gracefully(self) -> None:
        def mock_llm(prompt: str) -> str:
            return "I am unable to determine this because the server was disconnected halfway."

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Question", self.sample_chunks)

        self.assertIsInstance(res, StructuredSufficiencyResult)
        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.0)
        self.assertTrue(len(res.missing_information) > 0)

    def test_empty_string_response(self) -> None:
        def mock_llm(prompt: str) -> str:
            return "   \n\t  "

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        res = autorater.evaluate("Question", self.sample_chunks)
        self.assertFalse(res.sufficient)
        self.assertEqual(res.probability, 0.0)


class TestLLMSufficiencyAutoRaterFallbackBehavior(unittest.TestCase):
    """Test suite for fallback behavior to the deterministic judge."""

    def setUp(self) -> None:
        self.sample_chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech develops cloud-based analytics software. NovaTech was founded by Priya Mehta.",
                token_count=14,
                char_count=87,
            )
        ]

    def test_fallback_when_llm_raises_network_error(self) -> None:
        def broken_llm(prompt: str) -> str:
            raise ConnectionError("Remote LLM service connection refused")

        evaluator = ContextSufficiencyEvaluator(
            llm_client=broken_llm,
            fallback_to_heuristic=True,
        )

        # Fallback should catch ConnectionError and evaluate deterministically
        res = evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 1.0)
        self.assertEqual(res.missing_information, [])

    def test_fallback_when_no_llm_provided_in_evaluator(self) -> None:
        evaluator = ContextSufficiencyEvaluator()
        self.assertIsNone(evaluator.autorater)

        # Sufficient query
        res_suff = evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)
        self.assertTrue(res_suff.sufficient)
        self.assertEqual(res_suff.probability, 1.0)

        # Insufficient query
        res_insuff = evaluator.evaluate("Who is the Chief Financial Officer of NovaTech?", self.sample_chunks)
        self.assertFalse(res_insuff.sufficient)
        self.assertEqual(res_insuff.probability, 0.0)

    def test_evaluator_raises_when_fallback_disabled(self) -> None:
        def broken_llm(prompt: str) -> str:
            raise ConnectionError("Remote LLM service down")

        evaluator = ContextSufficiencyEvaluator(
            llm_client=broken_llm,
            fallback_to_heuristic=False,
        )

        with self.assertRaises(ConnectionError):
            evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)

    def test_standalone_autorater_fallback_judge(self) -> None:
        def broken_llm(prompt: str) -> str:
            raise RuntimeError("API quota exhausted")

        autorater = LLMSufficiencyAutoRater(
            llm_client=broken_llm,
            fallback_judge=AutoraterStyleSufficiencyJudge(),
        )

        res = autorater.evaluate("What does NovaTech develop?", self.sample_chunks)
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 1.0)

    def test_pipeline_integration_with_autorater(self) -> None:
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "sufficient": True,
                "probability": 0.99,
                "missing_information": [],
                "rationale": "Verified by pluggable AutoRater.",
            })

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        pipeline = EvidenceFirstPipeline(chunks=self.sample_chunks, autorater=autorater)
        trace = pipeline.run("What does NovaTech develop?")

        self.assertEqual(trace.final_decision, "ANSWERED")
        self.assertIsNotNone(trace.sufficiency_result)
        self.assertEqual(trace.sufficiency_result.rationale, "Verified by pluggable AutoRater.")
        self.assertEqual(trace.sufficiency_result.probability, 0.99)

    def test_pipeline_integration_with_broken_autorater_falls_back(self) -> None:
        def broken_llm(prompt: str) -> str:
            raise TimeoutError("LLM request timed out")

        evaluator = ContextSufficiencyEvaluator(llm_client=broken_llm, fallback_to_heuristic=True)
        pipeline = EvidenceFirstPipeline(chunks=self.sample_chunks, evaluator=evaluator)
        trace = pipeline.run("What does NovaTech develop?")

        self.assertEqual(trace.final_decision, "ANSWERED")
        self.assertIsNotNone(trace.sufficiency_result)
        self.assertTrue(trace.sufficiency_result.sufficient)


class TestLLMClientCredentialsAndRuntimeIntegration(unittest.TestCase):
    """Test suite for credential detection, provider selection, and runtime mode indicators."""

    def setUp(self) -> None:
        self.sample_chunks = [
            Chunk(
                chunk_id="c1",
                document_id="company",
                document_name="company.txt",
                page=1,
                section="Overview",
                text="NovaTech develops cloud-based analytics software.",
                token_count=8,
                char_count=52,
            )
        ]
        # Preserve original environment
        self.orig_gemini = os.environ.get("GEMINI_API_KEY")
        self.orig_google = os.environ.get("GOOGLE_API_KEY")
        self.orig_openai = os.environ.get("OPENAI_API_KEY")
        self.orig_gemini_model = os.environ.get("GEMINI_MODEL")
        self.orig_openai_model = os.environ.get("OPENAI_MODEL")
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("GEMINI_MODEL", None)
        os.environ.pop("OPENAI_MODEL", None)

    def tearDown(self) -> None:
        if self.orig_gemini is not None:
            os.environ["GEMINI_API_KEY"] = self.orig_gemini
        else:
            os.environ.pop("GEMINI_API_KEY", None)

        if self.orig_google is not None:
            os.environ["GOOGLE_API_KEY"] = self.orig_google
        else:
            os.environ.pop("GOOGLE_API_KEY", None)

        if self.orig_openai is not None:
            os.environ["OPENAI_API_KEY"] = self.orig_openai
        else:
            os.environ.pop("OPENAI_API_KEY", None)

        if self.orig_gemini_model is not None:
            os.environ["GEMINI_MODEL"] = self.orig_gemini_model
        else:
            os.environ.pop("GEMINI_MODEL", None)

        if self.orig_openai_model is not None:
            os.environ["OPENAI_MODEL"] = self.orig_openai_model
        else:
            os.environ.pop("OPENAI_MODEL", None)

    def test_create_llm_client_missing_credentials_returns_none(self) -> None:
        from agentic_rag.adapters.llm_client import create_llm_client

        client, desc = create_llm_client()
        self.assertIsNone(client)
        self.assertIn("Deterministic fallback", desc)

    def test_create_llm_client_discovers_gemini_key(self) -> None:
        from agentic_rag.adapters.llm_client import create_llm_client

        os.environ["GEMINI_API_KEY"] = "mock-gemini-key-12345"
        client, desc = create_llm_client()
        self.assertIsNotNone(client)
        self.assertEqual(client.provider, "gemini")
        self.assertEqual(client.model, "gemini-1.5-flash")
        self.assertIn("Gemini", desc)

    def test_create_llm_client_discovers_openai_key(self) -> None:
        from agentic_rag.adapters.llm_client import create_llm_client

        os.environ["OPENAI_API_KEY"] = "mock-openai-key-67890"
        client, desc = create_llm_client()
        self.assertIsNotNone(client)
        self.assertEqual(client.provider, "openai")
        self.assertEqual(client.model, "gpt-4o-mini")
        self.assertIn("OpenAI", desc)

    def test_evaluator_mode_recorded_when_llm_used(self) -> None:
        def mock_llm(prompt: str) -> str:
            return json.dumps({
                "sufficient": True,
                "probability": 0.97,
                "missing_information": [],
                "rationale": "LLM verified context completeness.",
            })

        autorater = LLMSufficiencyAutoRater(llm_client=mock_llm)
        autorater.mode_description = "LLM AutoRater (Gemini - gemini-1.5-flash)"
        evaluator = ContextSufficiencyEvaluator(autorater=autorater)
        res = evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)

        self.assertEqual(res.evaluator_mode, "LLM AutoRater (Gemini - gemini-1.5-flash)")
        self.assertTrue(res.sufficient)
        self.assertEqual(res.to_dict()["evaluator_mode"], "LLM AutoRater (Gemini - gemini-1.5-flash)")

        # Verify structured JSON output remains strictly valid
        json_str = res.to_json()
        parsed = json.loads(json_str)
        self.assertTrue(parsed["sufficient"])
        self.assertEqual(parsed["probability"], 0.97)
        self.assertEqual(parsed["missing_information"], [])
        self.assertIn("LLM verified", parsed["rationale"])

    def test_evaluator_mode_recorded_when_fallback_used(self) -> None:
        evaluator = ContextSufficiencyEvaluator()
        res = evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)

        self.assertEqual(res.evaluator_mode, "Deterministic fallback")
        self.assertTrue(res.sufficient)
        self.assertEqual(res.probability, 1.0)

    def test_evaluator_mode_recorded_when_llm_fails_and_falls_back(self) -> None:
        def failing_llm(prompt: str) -> str:
            raise urllib.error.URLError("Network unreachable")

        evaluator = ContextSufficiencyEvaluator(llm_client=failing_llm, fallback_to_heuristic=True)
        res = evaluator.evaluate("What does NovaTech develop?", self.sample_chunks)

    def test_no_secrets_toml_and_no_env_api_key_does_not_raise(self) -> None:
        """Regression test: No secrets.toml + no environment API key must not raise an exception."""
        import sys
        import types
        from agentic_rag.adapters.llm_client import create_llm_client, get_llm_credentials

        # Simulate Streamlit's StreamlitSecretNotFoundError when secrets.toml is absent
        class StreamlitSecretNotFoundError(FileNotFoundError):
            pass

        class MockStreamlitSecretsProxy:
            def get(self, key: str, default=None):
                raise StreamlitSecretNotFoundError(
                    "No secrets found. Valid paths for a secrets.toml file are: ~/.streamlit/secrets.toml"
                )

            def __getitem__(self, key: str):
                raise StreamlitSecretNotFoundError(
                    "No secrets found. Valid paths for a secrets.toml file are: ~/.streamlit/secrets.toml"
                )

        mock_st = types.ModuleType("streamlit")
        mock_st.secrets = MockStreamlitSecretsProxy()
        orig_streamlit = sys.modules.get("streamlit")
        sys.modules["streamlit"] = mock_st

        try:
            # Must not raise StreamlitSecretNotFoundError or FileNotFoundError
            provider, api_key, model = get_llm_credentials()
            self.assertIsNone(provider)
            self.assertIsNone(api_key)
            self.assertIsNone(model)

            client, desc = create_llm_client()
            self.assertIsNone(client)
            self.assertIn("Deterministic fallback", desc)
        finally:
            if orig_streamlit is not None:
                sys.modules["streamlit"] = orig_streamlit
            else:
                sys.modules.pop("streamlit", None)

    def test_streamlit_secrets_used_when_secrets_exist(self) -> None:
        """Verify secrets are retrieved when Streamlit secrets.toml actually exists."""
        import sys
        import types
        from agentic_rag.adapters.llm_client import create_llm_client, get_llm_credentials

        mock_st = types.ModuleType("streamlit")
        mock_st.secrets = {"GEMINI_API_KEY": "streamlit-secret-gemini-key"}
        orig_streamlit = sys.modules.get("streamlit")
        sys.modules["streamlit"] = mock_st

        try:
            provider, api_key, model = get_llm_credentials()
            self.assertEqual(provider, "gemini")
            self.assertEqual(api_key, "streamlit-secret-gemini-key")

            client, desc = create_llm_client()
            self.assertIsNotNone(client)
            self.assertIn("Gemini", desc)
        finally:
            if orig_streamlit is not None:
                sys.modules["streamlit"] = orig_streamlit
            else:
                sys.modules.pop("streamlit", None)


if __name__ == "__main__":
    unittest.main()
