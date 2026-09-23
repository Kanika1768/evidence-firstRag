"""Dependency-free sufficient-context judging utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from agentic_rag.contracts import (
    AnswerabilityLabel,
    AnswerStatus,
    ConflictEvidence,
    ConflictingEvidenceGroup,
    ContextAssessment,
    ContextStatus,
    CoveredFact,
    DraftAnswer,
    FeedbackQuery,
    GroundedAnswer,
    RequiredFact,
    RetrievalPlan,
    Route,
    Snippet,
)


@dataclass(frozen=True)
class AutoraterStyleSufficiencyJudge:
    """Heuristic judge shaped like a Sufficient Context autorater.

    The judge is deterministic and dependency-free so tests and offline demos can
    exercise answerability behavior without calling a model. Fact support is
    controlled by `RequiredFact.metadata["required_terms"]`; when absent, terms
    are derived from the fact description.
    """

    def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        draft: DraftAnswer,
    ) -> ContextAssessment:
        del question
        snippet_ids = {snippet.id for snippet in snippets}
        unsupported = tuple(
            claim.text
            for claim in draft.claims
            if not claim.snippet_ids or any(snippet_id not in snippet_ids for snippet_id in claim.snippet_ids)
        )

        covered: list[CoveredFact] = []
        missing: list[RequiredFact] = []
        conflicts: list[ConflictEvidence] = []

        for fact in plan.required_facts:
            route = plan.route_for_fact(fact.id)
            supporting_snippets = tuple(
                snippet
                for snippet in snippets
                if _snippet_supports_fact(snippet, fact, route)
            )
            if supporting_snippets:
                covered.append(CoveredFact(fact.id, tuple(snippet.id for snippet in supporting_snippets)))
                conflict = _conflicting_evidence(fact.id, supporting_snippets, fact.metadata)
                if conflict is not None:
                    conflicts.append(conflict)
            elif fact.is_must:
                missing.append(fact)

        must_fact_ids = {fact.id for fact in plan.required_facts if fact.is_must}
        must_count = len(must_fact_ids)
        covered_must_count = sum(1 for fact in covered if fact.fact_id in must_fact_ids)
        score = min(covered_must_count / must_count, 1.0) if must_count else 1.0

        feedback_queries = tuple(
            feedback
            for fact in missing
            for feedback in (_feedback_query(fact, plan),)
            if feedback is not None
        )
        missing_descriptions = tuple(fact.description for fact in missing)

        if conflicts:
            conflicting_fact_ids = tuple(conflict.fact_id for conflict in conflicts)
            return ContextAssessment(
                status=ContextStatus.INSUFFICIENT,
                sufficiency_score=min(score, 0.5),
                covered_facts=tuple(covered),
                missing_facts=missing_descriptions,
                unsupported_claims=unsupported,
                conflicts=tuple(conflicts),
                feedback_queries=feedback_queries,
                reason=f"Found conflicting evidence for required facts: {', '.join(conflicting_fact_ids)}.",
                answerability=AnswerabilityLabel.CONFLICTING,
            )

        if not missing and not unsupported:
            return ContextAssessment(
                status=ContextStatus.SUFFICIENT,
                sufficiency_score=score,
                covered_facts=tuple(covered),
                missing_facts=(),
                unsupported_claims=(),
                feedback_queries=(),
                reason="All required facts are supported by retrieved snippets.",
                answerability=AnswerabilityLabel.SUFFICIENT,
            )

        if missing and not snippets and not feedback_queries:
            return ContextAssessment(
                status=ContextStatus.UNANSWERABLE,
                sufficiency_score=0.0,
                covered_facts=tuple(covered),
                missing_facts=missing_descriptions,
                unsupported_claims=unsupported,
                feedback_queries=(),
                reason="Required facts have no retrieved evidence and no routed corpus for follow-up retrieval.",
                answerability=AnswerabilityLabel.UNANSWERABLE,
            )

        if missing and covered_must_count and not unsupported:
            return ContextAssessment(
                status=ContextStatus.INSUFFICIENT,
                sufficiency_score=score,
                covered_facts=tuple(covered),
                missing_facts=missing_descriptions,
                unsupported_claims=(),
                feedback_queries=feedback_queries,
                reason="Retrieved context covers some required facts but not enough for a definitive answer.",
                answerability=AnswerabilityLabel.USEFUL_BUT_INCOMPLETE,
            )

        return ContextAssessment(
            status=ContextStatus.INSUFFICIENT,
            sufficiency_score=min(score, 0.5),
            covered_facts=tuple(covered),
            missing_facts=missing_descriptions,
            unsupported_claims=unsupported,
            feedback_queries=feedback_queries,
            reason="Retrieved context is not sufficient for a grounded answer.",
            answerability=AnswerabilityLabel.INSUFFICIENT,
        )


ANSWERABILITY_STATUS_POLICY = {
    AnswerabilityLabel.SUFFICIENT: AnswerStatus.ANSWERED,
    AnswerabilityLabel.USEFUL_BUT_INCOMPLETE: AnswerStatus.PARTIAL,
    AnswerabilityLabel.CONFLICTING: AnswerStatus.PARTIAL,
    AnswerabilityLabel.INSUFFICIENT: AnswerStatus.UNANSWERABLE,
    AnswerabilityLabel.UNANSWERABLE: AnswerStatus.UNANSWERABLE,
}


def apply_selective_abstention_policy(
    answer: GroundedAnswer,
    assessment: ContextAssessment,
) -> GroundedAnswer:
    """Map answerability labels to answer statuses before final output."""

    target_status = _target_answer_status(assessment)
    if target_status == AnswerStatus.ANSWERED:
        return answer
    if target_status == AnswerStatus.PARTIAL:
        return GroundedAnswer(
            answer=_partial_answer_text(answer, assessment),
            citations=tuple(answer.citations),
            status=AnswerStatus.PARTIAL,
            missing_facts=tuple(answer.missing_facts) or tuple(assessment.missing_facts),
            sufficiency_score=assessment.sufficiency_score,
            conflicts=tuple(answer.conflicts) or tuple(assessment.conflicts),
        )
    return GroundedAnswer(
        answer="No grounded answer is available from the retrieved context.",
        citations=(),
        status=AnswerStatus.UNANSWERABLE,
        missing_facts=tuple(answer.missing_facts) or tuple(assessment.missing_facts),
        sufficiency_score=assessment.sufficiency_score,
        conflicts=tuple(answer.conflicts) or tuple(assessment.conflicts),
    )


def _snippet_supports_fact(snippet: Snippet, fact: RequiredFact, route: Route | None) -> bool:
    if route and snippet.corpus_id not in route.candidate_corpus_ids:
        return False
    required_terms = _metadata_terms(fact.metadata, "required_terms")
    if not required_terms:
        required_terms = tuple(_tokens(fact.description))
    text = snippet.text.lower()
    return all(str(term).lower() in text for term in required_terms)


def _conflicting_evidence(
    fact_id: str,
    snippets: Sequence[Snippet],
    metadata: Mapping[str, object],
) -> ConflictEvidence | None:
    conflict_terms = _metadata_terms(metadata, "conflict_terms")
    if not conflict_terms:
        return None
    groups = tuple(
        ConflictingEvidenceGroup(
            label=str(term).lower(),
            snippet_ids=tuple(
                snippet.id
                for snippet in snippets
                if str(term).lower() in snippet.text.lower()
            ),
            value=str(term).lower(),
        )
        for term in conflict_terms
    )
    groups = tuple(group for group in groups if group.snippet_ids)
    if len(groups) < 2:
        return None
    return ConflictEvidence(
        fact_id=fact_id,
        groups=groups,
        reason=f"Multiple incompatible values were found for required fact '{fact_id}'.",
    )


def _feedback_query(fact: RequiredFact, plan: RetrievalPlan) -> FeedbackQuery | None:
    route = plan.route_for_fact(fact.id)
    if route is None or not route.candidate_corpus_ids:
        return None
    recovery_terms = tuple(
        token
        for token in re.findall(
            r"[a-z0-9]+",
            fact.description.lower().replace("'s", ""),
        )
        if token not in {
            "what", "who", "when", "where", "why", "how",
            "does", "did", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "of", "to", "for",
            "in", "on", "by", "with",
        }
    )

    recovery_query = " ".join(dict.fromkeys(recovery_terms))

    return FeedbackQuery(
        query=recovery_query or fact.description,
        target_corpus_ids=tuple(route.candidate_corpus_ids),
        reason=f"Evidence for required fact '{fact.id}' was not found; reformulated recovery query.",
        fact_id=fact.id,
    )


def _metadata_terms(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _target_answer_status(assessment: ContextAssessment) -> AnswerStatus:
    if (
        assessment.answerability is None
        and _enum_value(assessment.status) == ContextStatus.INSUFFICIENT.value
        and assessment.covered_facts
    ):
        return AnswerStatus.PARTIAL
    return ANSWERABILITY_STATUS_POLICY[assessment.answerability_label]


def _partial_answer_text(answer: GroundedAnswer, assessment: ContextAssessment) -> str:
    if assessment.answerability_label == AnswerabilityLabel.CONFLICTING:
        return "Conflicting evidence prevents a definitive grounded answer."
    if answer.status == AnswerStatus.PARTIAL and answer.answer:
        return answer.answer
    if assessment.missing_facts:
        return "Partial evidence found; missing facts remain."
    return "Insufficient context for a definitive grounded answer."


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)
