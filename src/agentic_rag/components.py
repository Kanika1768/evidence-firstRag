"""Compatibility component layer for historical Agentic-RAG-Skill users.

This module keeps lightweight deterministic defaults while using the current
`agentic_rag` contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    ClaimCitation,
    ContextAssessment,
    CorpusDescriptor,
    CoveredFact,
    FeedbackQuery,
    GroundedAnswer,
    GroundedCitation,
    RequiredFact,
    RetrievalPlan,
    RetrievalRoute,
    RetrievalHit,
    SubQuery,
)


@dataclass(frozen=True)
class InMemoryDocument:
    """Simple document record for deterministic tests and demos."""

    id: str
    corpus_id: str
    text: str
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DraftAnswer:
    """Legacy draft container used by earlier scaffold versions."""

    text: str
    cited_snippet_ids: Sequence[str] = ()
    unsupported_claims: Sequence[str] = ()


class KeywordPlanner:
    """Route required facts to likely corpora by simple term overlap."""

    def create_plan(
        self,
        question: str,
        corpora: Sequence[CorpusDescriptor],
        prior_feedback: Sequence[FeedbackQuery] = (),
    ) -> RetrievalPlan:
        required_facts = tuple(_facts_from_question(question, prior_feedback))
        routes = tuple(
            RetrievalRoute(
                fact_id=fact.id,
                candidate_corpus_ids=_rank_corpora(fact.description, corpora),
                reason="Selected by overlap between the required fact and corpus descriptions.",
            )
            for fact in required_facts
        )
        return RetrievalPlan(
            question=question,
            required_facts=required_facts,
            routes=routes,
            stop_conditions=(
                "All must facts have cited snippets.",
                "No targeted follow-up queries remain.",
            ),
        )


class TemplateQueryRewriter:
    """Turn plan routes into readable, deterministic subqueries."""

    def rewrite(
        self,
        plan: RetrievalPlan,
        prior_feedback: Sequence[FeedbackQuery] = (),
    ) -> Sequence[SubQuery]:
        if prior_feedback:
            return tuple(
                SubQuery(
                    id=f"feedback-{index}",
                    fact_id=_feedback_fact_id(feedback, plan),
                    query=feedback.query,
                    target_corpus_ids=feedback.target_corpus_ids,
                    reason=feedback.reason,
                )
                for index, feedback in enumerate(prior_feedback, start=1)
            )

        fact_by_id = {fact.id: fact for fact in plan.required_facts}
        subqueries: list[SubQuery] = []
        for index, route in enumerate(plan.routes, start=1):
            fact = fact_by_id[route.fact_id]
            subqueries.append(
                SubQuery(
                    id=f"q{index}",
                    fact_id=fact.id,
                    query=fact.description,
                    target_corpus_ids=route.candidate_corpus_ids,
                    reason=route.reason,
                )
            )
        return tuple(subqueries)


class InMemoryKeywordRetriever:
    """Simple token-overlap retriever with deterministic scoring and provenance."""

    def __init__(self, documents: Sequence[InMemoryDocument], *, top_k: int = 3) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.documents = tuple(documents)
        self.top_k = top_k

    def retrieve(
        self,
        subqueries: Sequence[SubQuery],
        corpora: Sequence[CorpusDescriptor],
    ) -> Sequence[RetrievalHit]:
        _ = corpora
        valid_corpus_ids = {corpus.id for corpus in corpora}
        hits: list[RetrievalHit] = []
        for subquery in subqueries:
            query_tokens = _tokens(subquery.query)
            allowed = set(subquery.target_corpus_ids) & valid_corpus_ids
            scored = []
            for document in self.documents:
                if document.corpus_id not in allowed:
                    continue
                score = _score(query_tokens, _tokens(document.text))
                if score <= 0:
                    continue
                scored.append((score, document))
            scored.sort(key=lambda item: (-item[0], item[1].id))
            for rank, (score, document) in enumerate(scored[: self.top_k], start=1):
                hits.append(
                    RetrievalHit(
                        id=f"{subquery.id}:{document.id}:{rank}",
                        corpus_id=document.corpus_id,
                        document_id=document.id,
                        text=document.text,
                        score=score,
                        metadata={
                            "subquery_id": subquery.id,
                            "fact_id": subquery.fact_id,
                            **dict(document.metadata or {}),
                        },
                    )
                )
        return tuple(hits)


class CoverageSufficiencyJudge:
    """Require at least one routed snippet per must-have fact."""

    def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
        draft: DraftAnswer,
    ) -> ContextAssessment:
        del question
        hits_by_fact: dict[str, list[str]] = {}
        for hit in hits:
            fact_id = hit.metadata.get("fact_id")
            if isinstance(fact_id, str):
                hits_by_fact.setdefault(fact_id, []).append(hit.id)

        covered = tuple(
            CoveredFact(fact.id, tuple(hits_by_fact[fact.id]))
            for fact in plan.required_facts
            if hits_by_fact.get(fact.id)
        )
        missing = tuple(
            fact.description
            for fact in plan.required_facts
            if _enum_value(fact.priority) == "must" and not hits_by_fact.get(fact.id)
        )
        unsupported = tuple(getattr(draft, "unsupported_claims", ()))

        if not hits:
            return ContextAssessment(
                status="irrelevant",
                sufficiency_score=0.2,
                missing_facts=tuple(fact.description for fact in plan.required_facts if _enum_value(fact.priority) == "must"),
                reason="No retrieved snippets matched the routed subqueries.",
            )

        if missing or unsupported:
            return ContextAssessment(
                status="insufficient",
                sufficiency_score=0.45,
                covered_facts=covered,
                missing_facts=missing,
                unsupported_claims=unsupported,
                feedback_queries=tuple(_feedback_for_missing(plan, missing)),
                reason="At least one must-have fact lacks supporting snippets or draft has unsupported claims.",
            )

        return ContextAssessment(
            status="sufficient",
            sufficiency_score=0.85,
            covered_facts=covered,
            reason="Every must-have fact has at least one supporting snippet.",
        )


class ExtractiveSynthesizer:
    """Create a conservative final answer from retrieved snippets only."""

    def draft(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
    ) -> DraftAnswer:
        del question, plan
        lines = [f"[{hit.id}] {hit.text}" for hit in hits]
        return DraftAnswer(text="\n".join(lines), cited_snippet_ids=tuple(hit.id for hit in hits))

    def finalize(
        self,
        question: str,
        plan: RetrievalPlan,
        hits: Sequence[RetrievalHit],
        assessment: ContextAssessment,
        iterations: int,
    ) -> GroundedAnswer:
        del question, plan
        citations = tuple(
            GroundedCitation(
                claim=_claim_from_hit(hit),
                snippet_ids=(hit.id,),
            )
            for hit in hits
            if hit.id in _covered_snippet_ids(assessment)
        )
        if _enum_value(assessment.status) == "sufficient":
            answer = " ".join(f"{citation.claim} [{citation.snippet_ids[0]}]" for citation in citations)
            return GroundedAnswer(
                answer=answer,
                status="answered",
                citations=citations,
                iterations=iterations,
                sufficiency_score=assessment.sufficiency_score,
            )

        if _enum_value(assessment.status) == "unanswerable":
            status = "unanswerable"
            answer = "The available corpora do not contain enough evidence to answer."
        else:
            status = "partial"
            answer = "Partial evidence found. Missing facts: " + "; ".join(assessment.missing_facts)
        return GroundedAnswer(
            answer=answer,
            status=status,
            citations=citations,
            missing_facts=assessment.missing_facts,
            iterations=iterations,
            sufficiency_score=assessment.sufficiency_score,
        )


def _facts_from_question(question: str, prior_feedback: Sequence[FeedbackQuery]) -> list[RequiredFact]:
    pieces = [piece.strip(" ?.") for piece in re.split(r"\band\b|,|;", question, flags=re.IGNORECASE)]
    facts = [piece for piece in pieces if piece]
    if not facts:
        facts = [question.strip()]
    for feedback in prior_feedback:
        facts.append(feedback.query)
    deduped: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        key = _normalize(fact)
        if key and key not in seen:
            seen.add(key)
            deduped.append(fact)
    return [RequiredFact(id=f"f{index}", description=fact) for index, fact in enumerate(deduped, start=1)]


def _rank_corpora(fact: str, corpora: Sequence[CorpusDescriptor]) -> tuple[str, ...]:
    fact_tokens = _tokens(fact)
    ranked = sorted(
        ((_score(fact_tokens, _tokens(corpus.description)), corpus.id) for corpus in corpora),
        key=lambda item: (-item[0], item[1]),
    )
    selected = tuple(corpus_id for score, corpus_id in ranked if score > 0)
    if selected:
        return selected[:2]
    return (ranked[0][1],) if ranked else ()


def _score(query_tokens: set[str], text_tokens: set[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / len(query_tokens)


def _feedback_fact_id(feedback: FeedbackQuery, plan: RetrievalPlan) -> str:
    feedback_tokens = _tokens(feedback.query)
    best = max(
        plan.required_facts,
        key=lambda fact: _score(feedback_tokens, _tokens(fact.description)),
        default=None,
    )
    return best.id if best else "feedback"


def _feedback_for_missing(plan: RetrievalPlan, missing: Sequence[str]) -> list[FeedbackQuery]:
    routes_by_fact = {route.fact_id: route for route in plan.routes}
    feedback: list[FeedbackQuery] = []
    for fact in plan.required_facts:
        if fact.description not in missing:
            continue
        route = routes_by_fact.get(fact.id)
        feedback.append(
            FeedbackQuery(
                query=fact.description,
                target_corpus_ids=route.candidate_corpus_ids if route else (),
                reason=f"Search specifically for missing fact {fact.id}: {fact.description}",
                fact_id=fact.id,
            )
        )
    return feedback


def _covered_snippet_ids(assessment: ContextAssessment) -> set[str]:
    return {snippet_id for fact in assessment.covered_facts for snippet_id in fact.snippet_ids}


def _claim_from_hit(hit: RetrievalHit) -> str:
    text = " ".join(hit.text.split())
    return text if len(text) <= 180 else text[:177] + "..."


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)
