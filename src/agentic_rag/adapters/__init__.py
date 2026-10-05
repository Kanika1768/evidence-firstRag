"""Portable adapters for the Agentic RAG scaffold."""

from .in_memory import InMemoryDocument, RuleBasedSynthesizer, ScriptedPlanner, SnippetDrafter, EvidenceCoverageJudge, FeedbackAwareQueryRewriter
from .retriever import LexicalDocument, LexicalRetriever, LEXICAL_SCORING_RULE
from .llm import SchemaValidationError
from .llm_client import GenericRestLLMClient, create_llm_client, get_llm_credentials
from .structured_sufficiency import LLMSufficiencyAutoRater, StructuredSufficiencyResult
from .vertex_rag import VertexRagConfig, VertexRagCrossCorpusRetriever

__all__ = [
    "EvidenceCoverageJudge",
    "FeedbackAwareQueryRewriter",
    "GenericRestLLMClient",
    "InMemoryDocument",
    "LEXICAL_SCORING_RULE",
    "LLMSufficiencyAutoRater",
    "LexicalDocument",
    "LexicalRetriever",
    "RuleBasedSynthesizer",
    "SchemaValidationError",
    "ScriptedPlanner",
    "SnippetDrafter",
    "StructuredSufficiencyResult",
    "VertexRagConfig",
    "VertexRagCrossCorpusRetriever",
    "create_llm_client",
    "get_llm_credentials",
]
