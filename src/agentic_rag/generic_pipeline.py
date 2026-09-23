"""Generic document-grounded EvidenceFirst RAG pipeline."""

from __future__ import annotations

import re

from typing import Sequence

from .contracts import (
    Claim,
    ContextAssessment,
    Corpus,
    DraftAnswer,
    GroundedAnswer,
    GroundedCitation,
    RetrievalPlan,
    Snippet,
    Subquery,
)
from .general_planner import GeneralPlanner
from .sufficiency import AutoraterStyleSufficiencyJudge


class GenericQueryRewriter:
    """Convert required facts and recovery feedback into retrieval queries."""

    def rewrite(
        self,
        question: str,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> Sequence[Subquery]:
        del question

        queries: list[Subquery] = []

        # First iteration: one query per required fact.
        if prior_assessment is None:
            for index, fact in enumerate(plan.required_facts, start=1):
                route = plan.route_for_fact(fact.id)

                queries.append(
                    Subquery(
                        id=f"q{index}",
                        fact_id=fact.id,
                        query=fact.description,
                        target_corpus_ids=tuple(
                            route.candidate_corpus_ids if route else ()
                        ),
                        reason="Retrieve evidence for the required fact.",
                        iteration=iteration,
                    )
                )

            return tuple(queries)

        # Recovery iteration: search specifically for missing facts.
        for index, feedback in enumerate(
            prior_assessment.feedback_queries,
            start=1,
        ):
            queries.append(
                Subquery(
                    id=f"recovery-{iteration}-{index}",
                    fact_id=feedback.fact_id or "recovery",
                    query=feedback.query,
                    target_corpus_ids=tuple(feedback.target_corpus_ids),
                    reason=feedback.reason,
                    parent_query=plan.question,
                    iteration=iteration,
                )
            )

        return tuple(queries)


class GenericExtractiveDrafter:
    """Create claims directly from retrieved evidence."""

    def draft(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
    ) -> DraftAnswer:
        del question

        claims: list[Claim] = []
        seen: set[str] = set()

        for fact in plan.required_facts:
            route = plan.route_for_fact(fact.id)

            for snippet in snippets:
                if route and snippet.corpus_id not in route.candidate_corpus_ids:
                    continue

                if snippet.fact_id != fact.id:
                    continue

                normalized = " ".join(snippet.text.lower().split())

                if normalized in seen:
                    continue

                seen.add(normalized)

                claims.append(
                    Claim(
                        text=" ".join(snippet.text.split()),
                        snippet_ids=(snippet.id,),
                    )
                )

        return DraftAnswer(
            claims=tuple(claims),
            cited_snippet_ids=tuple(
                snippet_id
                for claim in claims
                for snippet_id in claim.snippet_ids
            ),
        )


class GenericGroundedSynthesizer:
    """Produce concise answers using only evidence accepted by the judge."""

    def synthesize(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        assessment: ContextAssessment,
    ) -> GroundedAnswer:
        covered_ids = {
            snippet_id
            for covered_fact in assessment.covered_facts
            for snippet_id in covered_fact.snippet_ids
        }

        grounded_snippets = tuple(
            snippet
            for snippet in snippets
            if snippet.id in covered_ids
        )

        citations = tuple(
            GroundedCitation(
                claim=_best_evidence_sentence(question, snippet.text),
                snippet_ids=(snippet.id,),
            )
            for snippet in grounded_snippets
        )

        if assessment.answerability_label.value == "sufficient":
            answer_parts = tuple(
                citation.claim
                for citation in citations
                if citation.claim
            )

            answer = " ".join(
                dict.fromkeys(answer_parts)
            )

            return GroundedAnswer(
                answer=answer,
                citations=citations,
                status="answered",
                missing_facts=(),
                sufficiency_score=assessment.sufficiency_score,
                conflicts=tuple(assessment.conflicts),
            )

        if assessment.answerability_label.value == "conflicting":
            return GroundedAnswer(
                answer="Conflicting evidence prevents a definitive grounded answer.",
                citations=citations,
                status="partial",
                missing_facts=tuple(assessment.missing_facts),
                sufficiency_score=assessment.sufficiency_score,
                conflicts=tuple(assessment.conflicts),
            )

        return GroundedAnswer(
            answer="Insufficient evidence for a definitive grounded answer.",
            citations=citations,
            status="unanswerable",
            missing_facts=tuple(assessment.missing_facts),
            sufficiency_score=assessment.sufficiency_score,
            conflicts=tuple(assessment.conflicts),
        )


def _best_evidence_sentence(question: str, text: str) -> str:
    """Select the sentence containing the strongest question evidence."""

    sentences = tuple(
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    )

    if not sentences:
        return " ".join(text.split())

    question_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) > 2
        and token not in {
            "what", "who", "when", "where", "why", "how",
            "does", "did", "was", "were", "the", "and",
            "about", "from", "for", "with",
        }
    }

    def normalize(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                return token[:-len(suffix)]
        return token

    scored = []

    for index, sentence in enumerate(sentences):
        sentence_terms = {
            normalize(token)
            for token in re.findall(r"[a-z0-9]+", sentence.lower())
        }

        normalized_question_terms = {
            normalize(token)
            for token in question_terms
        }

        overlap = len(normalized_question_terms & sentence_terms)

        scored.append(
            (
                overlap,
                -index,
                sentence,
            )
        )

    scored.sort(reverse=True)

    return scored[0][2]

def build_generic_components():
    """Construct the dependency-free generic EvidenceFirst components."""

    return {
        "planner": GeneralPlanner(),
        "rewriter": GenericQueryRewriter(),
        "judge": AutoraterStyleSufficiencyJudge(),
        "drafter": GenericExtractiveDrafter(),
        "synthesizer": GenericGroundedSynthesizer(),
    }
