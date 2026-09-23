"""Protocol adapters for LLM-backed Agentic RAG roles."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from .contracts import (
    ContextAssessment,
    CorpusDescriptor,
    DraftAnswer,
    FeedbackQuery,
    GroundedAnswer,
    RetrievalHit,
    RetrievalPlan,
    SubQuery,
)
from .structured import (
    StructuredLLM,
    answer_from_mapping,
    assessment_from_mapping,
    draft_from_mapping,
    plan_from_mapping,
    subqueries_from_mapping,
)


class LLMPlanner:
    def __init__(self, llm: StructuredLLM, *, prompt: str) -> None:
        self.llm = llm
        self.prompt = prompt

    def create_plan(
        self,
        question: str,
        corpora: Sequence[CorpusDescriptor],
        prior_feedback: Sequence[FeedbackQuery] = (),
    ) -> RetrievalPlan:
        data = self.llm.complete_json(
            task="planner",
            prompt=self.prompt,
            payload={
                "question": question,
                "corpora": [asdict(corpus) for corpus in corpora],
                "prior_feedback": [asdict(feedback) for feedback in prior_feedback],
            },
        )
        return plan_from_mapping(data)


class LLMQueryRewriter:
    def __init__(self, llm: StructuredLLM, *, prompt: str) -> None:
        self.llm = llm
        self.prompt = prompt

    def rewrite(
        self,
        plan: RetrievalPlan,
        prior_feedback: Sequence[FeedbackQuery] = (),
    ) -> Sequence[SubQuery]:
        data = self.llm.complete_json(
            task="query_rewriter",
            prompt=self.prompt,
            payload={
                "plan": asdict(plan),
                "prior_feedback": [asdict(feedback) for feedback in prior_feedback],
            },
        )
        return subqueries_from_mapping(data)


class LLMSufficiencyJudge:
    def __init__(self, llm: StructuredLLM, *, prompt: str) -> None:
        self.llm = llm
        self.prompt = prompt

    def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
        draft: DraftAnswer,
    ) -> ContextAssessment:
        data = self.llm.complete_json(
            task="sufficiency_judge",
            prompt=self.prompt,
            payload={
                "question": question,
                "plan": asdict(plan),
                "hits": [asdict(hit) for hit in hits],
                "draft": asdict(draft),
            },
        )
        return assessment_from_mapping(data)


class LLMSynthesizer:
    def __init__(self, llm: StructuredLLM, *, draft_prompt: str, final_prompt: str) -> None:
        self.llm = llm
        self.draft_prompt = draft_prompt
        self.final_prompt = final_prompt

    def draft(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
    ) -> DraftAnswer:
        data = self.llm.complete_json(
            task="draft",
            prompt=self.draft_prompt,
            payload={
                "question": question,
                "plan": asdict(plan),
                "hits": [asdict(hit) for hit in hits],
            },
        )
        return draft_from_mapping(data)

    def finalize(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
        assessment: ContextAssessment,
        iterations: int,
    ) -> GroundedAnswer:
        data = self.llm.complete_json(
            task="synthesis",
            prompt=self.final_prompt,
            payload={
                "question": question,
                "plan": asdict(plan),
                "hits": [asdict(hit) for hit in hits],
                "assessment": asdict(assessment),
            },
        )
        return answer_from_mapping(data, iterations=iterations)
