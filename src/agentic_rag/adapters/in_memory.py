"""Deterministic in-memory adapters for tests, demos, and offline evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from agentic_rag.contracts import (
    AnswerabilityLabel,
    AnswerStatus,
    Claim,
    ContextAssessment,
    ContextStatus,
    Corpus,
    CoveredFact,
    DraftAnswer,
    FeedbackQuery,
    GroundedAnswer,
    GroundedCitation,
    RequiredFact,
    RetrievalPlan,
    Route,
    Snippet,
    Subquery,
)


@dataclass(frozen=True)
class InMemoryDocument:
    corpus_id: str
    document_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class ScriptedPlanner:
    def __init__(self, plan: RetrievalPlan) -> None:
        self.plan_result = plan

    def plan(
        self,
        question: str,
        corpus_catalog: Sequence[Corpus],
        prior_assessment: ContextAssessment | None = None,
    ) -> RetrievalPlan:
        return self.plan_result


class FeedbackAwareQueryRewriter:
    def __init__(self, initial_fact_ids: Sequence[str] | None = None) -> None:
        self.initial_fact_ids = tuple(initial_fact_ids) if initial_fact_ids is not None else None

    def rewrite(
        self,
        question: str,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> Sequence[Subquery]:
        if prior_assessment and prior_assessment.feedback_queries:
            return tuple(
                Subquery(
                    id=f"q{iteration}-{index}",
                    fact_id=feedback.fact_id or f"feedback-{index}",
                    query=feedback.query,
                    target_corpus_ids=tuple(feedback.target_corpus_ids),
                    reason=feedback.reason,
                    parent_query=question,
                    iteration=iteration,
                )
                for index, feedback in enumerate(prior_assessment.feedback_queries)
            )

        routes = tuple(
            route
            for route in plan.routes
            if self.initial_fact_ids is None or route.fact_id in self.initial_fact_ids
        )
        facts_by_id = {fact.id: fact for fact in plan.required_facts}
        return tuple(
            Subquery(
                id=f"q{iteration}-{index}",
                fact_id=route.fact_id,
                query=facts_by_id[route.fact_id].description,
                target_corpus_ids=tuple(route.candidate_corpus_ids),
                reason=route.reason,
                parent_query=question,
                iteration=iteration,
            )
            for index, route in enumerate(routes)
        )


class InMemoryRetriever:
    def __init__(self, documents: Sequence[InMemoryDocument], *, per_query_limit: int = 5) -> None:
        self.documents = tuple(documents)
        self.per_query_limit = per_query_limit
        self.calls: list[Subquery] = []

    def retrieve(self, subquery: Subquery) -> Sequence[Snippet]:
        self.calls.append(subquery)
        query_terms = _tokens(subquery.query)
        matches: list[Snippet] = []
        for document in self.documents:
            if document.corpus_id not in subquery.target_corpus_ids:
                continue
            text_terms = _tokens(document.text)
            overlap = query_terms & text_terms
            phrase_hit = subquery.query.lower() in document.text.lower()
            if not overlap and not phrase_hit:
                continue
            score = float(len(overlap)) + (2.0 if phrase_hit else 0.0)
            matches.append(
                Snippet(
                    id=f"{document.corpus_id}:{document.document_id}:{subquery.id}",
                    corpus_id=document.corpus_id,
                    document_id=document.document_id,
                    text=document.text,
                    score=score,
                    metadata=document.metadata,
                    query_id=subquery.id,
                    fact_id=subquery.fact_id,
                )
            )
        return tuple(sorted(matches, key=lambda snippet: snippet.score, reverse=True)[: self.per_query_limit])


class SnippetDrafter:
    def draft(self, question: str, plan: RetrievalPlan, snippets: Sequence[Snippet]) -> DraftAnswer:
        claims = tuple(Claim(snippet.text, (snippet.id,)) for snippet in snippets)
        return DraftAnswer(claims=claims, text="\n".join(claim.text for claim in claims))


class EvidenceCoverageJudge:
    def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        draft: DraftAnswer,
    ) -> ContextAssessment:
        snippet_ids = {snippet.id for snippet in snippets}
        unsupported = tuple(
            claim.text
            for claim in draft.claims
            if not claim.snippet_ids or any(snippet_id not in snippet_ids for snippet_id in claim.snippet_ids)
        )
        covered: list[CoveredFact] = []
        missing: list[RequiredFact] = []

        for fact in plan.required_facts:
            matching_snippets = tuple(
                snippet.id
                for snippet in snippets
                if self._snippet_supports_fact(snippet, fact, plan.route_for_fact(fact.id))
            )
            if matching_snippets:
                covered.append(CoveredFact(fact.id, matching_snippets))
            elif fact.is_must:
                missing.append(fact)

        must_fact_ids = {fact.id for fact in plan.required_facts if fact.is_must}
        must_count = len(must_fact_ids)
        covered_must_count = sum(1 for fact in covered if fact.fact_id in must_fact_ids)
        score = min(covered_must_count / must_count, 1.0) if must_count else 1.0
        status = ContextStatus.SUFFICIENT if not missing and not unsupported else ContextStatus.INSUFFICIENT
        feedback_queries = tuple(self._feedback_query(fact, plan) for fact in missing)
        reason = "All required facts are supported." if status == ContextStatus.SUFFICIENT else "Missing facts or unsupported draft claims remain."
        return ContextAssessment(
            status=status,
            sufficiency_score=score,
            covered_facts=tuple(covered),
            missing_facts=tuple(fact.description for fact in missing),
            unsupported_claims=unsupported,
            feedback_queries=feedback_queries,
            reason=reason,
        )

    def _snippet_supports_fact(self, snippet: Snippet, fact: RequiredFact, route: Route | None) -> bool:
        if route and snippet.corpus_id not in route.candidate_corpus_ids:
            return False
        required_terms = tuple(fact.metadata.get("required_terms", ())) if fact.metadata else ()
        if not required_terms:
            required_terms = tuple(_tokens(fact.description))
        text = snippet.text.lower()
        return all(str(term).lower() in text for term in required_terms)

    def _feedback_query(self, fact: RequiredFact, plan: RetrievalPlan) -> FeedbackQuery:
        route = plan.route_for_fact(fact.id)
        return FeedbackQuery(
            query=fact.description,
            target_corpus_ids=tuple(route.candidate_corpus_ids if route else ()),
            reason=f"Evidence for required fact '{fact.id}' was not found.",
            fact_id=fact.id,
        )


class RuleBasedSynthesizer:
    def synthesize(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        assessment: ContextAssessment,
    ) -> GroundedAnswer:
        snippets_by_id = {snippet.id: snippet for snippet in snippets}
        citations = tuple(
            GroundedCitation(
                claim=covered.fact_id,
                snippet_ids=tuple(snippet_id for snippet_id in covered.snippet_ids if snippet_id in snippets_by_id),
            )
            for covered in assessment.covered_facts
        )
        if assessment.answerability_label == AnswerabilityLabel.CONFLICTING and assessment.conflicts:
            conflict_citations = tuple(
                GroundedCitation(
                    claim=f"{conflict.fact_id}:{group.label}",
                    snippet_ids=tuple(
                        snippet_id for snippet_id in group.snippet_ids if snippet_id in snippets_by_id
                    ),
                )
                for conflict in assessment.conflicts
                for group in conflict.groups
            )
            conflict_citations = tuple(citation for citation in conflict_citations if citation.snippet_ids)
            return GroundedAnswer(
                answer="Conflicting evidence prevents a definitive grounded answer.",
                citations=conflict_citations,
                status=AnswerStatus.PARTIAL,
                missing_facts=tuple(assessment.missing_facts),
                sufficiency_score=assessment.sufficiency_score,
                conflicts=tuple(assessment.conflicts),
            )

        if _enum_value(assessment.status) == ContextStatus.SUFFICIENT.value:
            answer_lines = [
                f"{citation.claim}: " + " ".join(f"[{snippet_id}]" for snippet_id in citation.snippet_ids)
                for citation in citations
            ]
            return GroundedAnswer(
                answer="\n".join(answer_lines),
                citations=citations,
                status=AnswerStatus.ANSWERED,
                missing_facts=(),
                sufficiency_score=assessment.sufficiency_score,
            )

        status = AnswerStatus.PARTIAL if citations else AnswerStatus.UNANSWERABLE
        return GroundedAnswer(
            answer="Partial evidence found." if citations else "No grounded answer is available from the retrieved context.",
            citations=citations,
            status=status,
            missing_facts=tuple(assessment.missing_facts),
            sufficiency_score=assessment.sufficiency_score,
            conflicts=tuple(assessment.conflicts),
        )


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)
