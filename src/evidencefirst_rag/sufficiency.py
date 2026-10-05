"""Context Sufficiency Evaluator (Person 3 — Kanika).

Evaluates whether retrieved evidence is strictly sufficient for a diligent reader
to answer the question without external knowledge.
Outputs validated JSON with probability score, missing facts, and rationale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from agentic_rag.adapters.structured_sufficiency import (
    LLMSufficiencyAutoRater,
    StructuredSufficiencyResult,
)
from agentic_rag.prompts.sufficiency_v1 import (
    SUFFICIENCY_PROMPT_VERSION,
    build_sufficiency_prompt,
    get_sufficiency_json_schema,
    get_sufficiency_system_prompt,
)
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge
from evidencefirst_rag.chunking import Chunk

__all__ = [
    "ContextSufficiencyEvaluator",
    "LLMSufficiencyAutoRater",
    "StructuredSufficiencyResult",
]


class ContextSufficiencyEvaluator:
    """Evaluates context sufficiency under the Diligent Reader standard."""

    def __init__(
        self,
        prompt_version: str = SUFFICIENCY_PROMPT_VERSION,
        autorater: LLMSufficiencyAutoRater | None = None,
        llm_client: Any | None = None,
        fallback_to_heuristic: bool = True,
    ) -> None:
        self.prompt_version = prompt_version
        self.autorater_judge = AutoraterStyleSufficiencyJudge()
        self.fallback_to_heuristic = fallback_to_heuristic

        if autorater is not None:
            self.autorater = autorater
        elif llm_client is not None:
            self.autorater = LLMSufficiencyAutoRater(
                llm_client=llm_client,
                prompt_version=prompt_version,
                fallback_judge=None,
                catch_exceptions=False,
            )
        else:
            self.autorater = None

    def evaluate(
        self,
        question: str,
        chunks: Sequence[Chunk],
    ) -> StructuredSufficiencyResult:
        """Evaluate sufficiency of retrieved chunks for the question."""
        if not chunks:
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.0,
                missing_information=[question],
                rationale="No evidence chunks were retrieved.",
            )

        if self.autorater is not None:
            try:
                res = self.autorater.evaluate(question, chunks)
                if res is not None:
                    res.validate()
                    return res
            except Exception:
                if not self.fallback_to_heuristic:
                    raise
                # Fall through to deterministic heuristic fallback

        res = self._evaluate_heuristic(question, chunks)
        res.evaluator_mode = "Deterministic fallback"
        return res

    def _evaluate_heuristic(
        self,
        question: str,
        chunks: Sequence[Chunk],
    ) -> StructuredSufficiencyResult:

        combined_text = "\n\n".join(f"[{c.document_name}, Page {c.page}]: {c.text}" for c in chunks)

        from agentic_rag.contracts import (
            Claim,
            Corpus,
            DraftAnswer,
            FactPriority,
            RequiredFact,
            RetrievalPlan,
            Route,
            Snippet,
        )
        from agentic_rag.general_planner import GeneralPlanner, _meaningful_terms

        unique_doc_ids = sorted({c.document_id for c in chunks})
        corpora = tuple(
            Corpus(id=doc_id, description=f"Document {doc_id}")
            for doc_id in unique_doc_ids
        )

        planner = GeneralPlanner()
        plan = planner.plan(question, corpora)

        if not plan.required_facts:
            fact = RequiredFact(
                id="f1",
                description=question,
                priority=FactPriority.MUST,
                metadata={"required_terms": _meaningful_terms(question)},
            )
            route = Route(
                fact_id="f1",
                candidate_corpus_ids=tuple(unique_doc_ids),
                reason="Default route for question",
            )
            plan = RetrievalPlan(
                question=question,
                required_facts=(fact,),
                routes=(route,),
            )

        snippets = tuple(
            Snippet(id=c.chunk_id, corpus_id=c.document_id, document_id=c.document_id, text=c.text)
            for c in chunks
        )
        claims = tuple(Claim(text=c.text, snippet_ids=(c.chunk_id,)) for c in chunks)
        draft = DraftAnswer(text=combined_text, claims=claims)

        assessment = self.autorater_judge.assess(question, plan, snippets, draft)
        result = StructuredSufficiencyResult.from_context_assessment(assessment)
        result.validate()
        return result

    def evaluate_from_json(self, json_str: str) -> StructuredSufficiencyResult:
        """Parse raw model output string with robust repair fallback."""
        return StructuredSufficiencyResult.from_json(json_str)
