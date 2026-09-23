"""Orchestrator for plan, retrieve, assess, iterate, and synthesize."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Sequence

from .contracts import (
    AnswerStatus,
    ContextAssessment,
    ContextStatus,
    Corpus,
    Drafter,
    GroundedAnswer,
    GroundedCitation,
    IterationState,
    IterationTrace,
    Planner,
    QueryRewriter,
    Retriever,
    RunResult,
    Snippet,
    SufficientContextJudge,
    Synthesizer,
)
from .sufficiency import apply_selective_abstention_policy


@dataclass(frozen=True)
class OrchestratorConfig:
    max_iterations: int = 3
    max_snippets: int = 100

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.max_snippets < 1:
            raise ValueError("max_snippets must be at least 1")


class AgenticRAGOrchestrator:
    def __init__(
        self,
        *,
        planner: Planner,
        rewriter: QueryRewriter,
        retriever: Retriever,
        drafter: Drafter,
        judge: SufficientContextJudge,
        synthesizer: Synthesizer,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.planner = planner
        self.rewriter = rewriter
        self.retriever = retriever
        self.drafter = drafter
        self.judge = judge
        self.synthesizer = synthesizer
        self.config = config or OrchestratorConfig()

    def run(self, question: str, corpus_catalog: Sequence[Corpus]) -> RunResult:
        plan = self.planner.plan(question, tuple(corpus_catalog), prior_assessment=None)
        prior_assessment: ContextAssessment | None = None
        snippets_by_id: dict[str, Snippet] = {}
        traces: list[IterationTrace] = []

        for iteration in range(self.config.max_iterations):
            subqueries = tuple(self.rewriter.rewrite(question, plan, prior_assessment, iteration))
            retrieved = self._retrieve_and_dedupe(subqueries, snippets_by_id)
            draft = self.drafter.draft(question, plan, tuple(snippets_by_id.values()))
            assessment = self.judge.assess(question, plan, tuple(snippets_by_id.values()), draft)
            traces.append(
                IterationTrace(
                    iteration=iteration,
                    subqueries=subqueries,
                    snippets=retrieved,
                    draft=draft,
                    assessment=assessment,
                )
            )

            if _enum_value(assessment.status) in {
                ContextStatus.SUFFICIENT.value,
                ContextStatus.IRRELEVANT.value,
                ContextStatus.UNANSWERABLE.value,
            }:
                break
            if not subqueries:
                break

            prior_assessment = assessment

        final_assessment = traces[-1].assessment
        answer = self.synthesizer.synthesize(
            question,
            plan,
            tuple(snippets_by_id.values()),
            final_assessment,
        )
        answer = self._guard_answer(answer, final_assessment, tuple(snippets_by_id.values()))
        return RunResult(
            question=question,
            plan=plan,
            answer=answer,
            iterations=tuple(traces),
            snippets=tuple(snippets_by_id.values()),
        )

    def _retrieve_and_dedupe(
        self,
        subqueries: Sequence,
        snippets_by_id: dict[str, Snippet],
    ) -> tuple[Snippet, ...]:
        retrieved: list[Snippet] = []
        for subquery in subqueries:
            for snippet in self.retriever.retrieve(subquery):
                if snippet.id in snippets_by_id:
                    continue
                if len(snippets_by_id) >= self.config.max_snippets:
                    return tuple(retrieved)
                snippets_by_id[snippet.id] = snippet
                retrieved.append(snippet)
        return tuple(retrieved)

    def _guard_answer(
        self,
        answer: GroundedAnswer,
        assessment: ContextAssessment,
        snippets: tuple[Snippet, ...],
    ) -> GroundedAnswer:
        answer = apply_selective_abstention_policy(answer, assessment)

        known_snippet_ids = {snippet.id for snippet in snippets}
        grounded_citations = tuple(
            GroundedCitation(
                claim=citation.claim,
                snippet_ids=tuple(
                    snippet_id for snippet_id in citation.snippet_ids if snippet_id in known_snippet_ids
                ),
            )
            for citation in answer.citations
            if any(snippet_id in known_snippet_ids for snippet_id in citation.snippet_ids)
        )
        if len(grounded_citations) != len(tuple(answer.citations)):
            return GroundedAnswer(
                answer=answer.answer,
                citations=grounded_citations,
                status=AnswerStatus.PARTIAL,
                missing_facts=tuple(answer.missing_facts) or tuple(assessment.missing_facts),
                sufficiency_score=assessment.sufficiency_score,
                conflicts=tuple(answer.conflicts) or tuple(assessment.conflicts),
            )
        return answer


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)


class AgenticRAGPipeline:
    """Backward-compatible pipeline API for projects relying on the legacy scaffold."""

    def __init__(
        self,
        *,
        planner,
        query_rewriter,
        retriever,
        sufficiency_judge,
        synthesizer,
        max_iterations: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        self.planner = planner
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.sufficiency_judge = sufficiency_judge
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations

    def answer(self, question: str, corpora: Sequence[Corpus]) -> GroundedAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not corpora:
            raise ValueError("at least one corpus is required")
        _validate_corpora(corpora)

        all_hits: list[Snippet] = []
        prior_feedback: tuple = ()
        audit_iterations: list[dict] = []
        last_state: IterationState | None = None
        last_assessment: ContextAssessment | None = None

        for iteration in range(1, self.max_iterations + 1):
            if hasattr(self.planner, "create_plan"):
                plan = self.planner.create_plan(question, corpora, prior_feedback)
            elif hasattr(self.planner, "plan"):
                plan = self.planner.plan(question, corpora, prior_feedback)
            else:
                raise TypeError("planner must expose create_plan(question, corpora, prior_feedback)")

            _validate_plan(plan, corpora)

            if hasattr(self.query_rewriter, "rewrite"):
                subqueries = tuple(self.query_rewriter.rewrite(plan, prior_feedback))
            else:
                raise TypeError("query_rewriter must expose rewrite(plan, prior_feedback)")

            _validate_subqueries(subqueries, plan, corpora)

            if hasattr(self.retriever, "retrieve"):
                new_hits = tuple(self.retriever.retrieve(subqueries, corpora))
            else:
                raise TypeError("retriever must expose retrieve(subqueries, corpora)")

            _validate_hits(new_hits, corpora)
            all_hits = _deduplicate_hits([*all_hits, *new_hits])
            draft = self.synthesizer.draft(question, plan, all_hits)
            assessment = self.sufficiency_judge.assess(question, plan, all_hits, draft)
            last_assessment = assessment
            _validate_assessment(assessment, plan, all_hits, corpora)

            state = IterationState(
                question=question,
                iteration=iteration,
                plan=plan,
                subqueries=subqueries,
                hits=tuple(all_hits),
                draft=draft,
                assessment=assessment,
                prior_feedback=prior_feedback,
            )
            last_state = state
            audit_iterations.append(_state_to_audit(state))

            if _enum_value(assessment.status) in {"sufficient", "unanswerable", "irrelevant"}:
                answer = self.synthesizer.finalize(question, plan, tuple(all_hits), assessment, iteration)
                return _with_audit(_guard_answer(answer, assessment, tuple(all_hits)), audit_iterations)

            prior_feedback = assessment.feedback_queries
            if not prior_feedback:
                answer = self.synthesizer.finalize(question, plan, tuple(all_hits), assessment, iteration)
                return _with_audit(_guard_answer(answer, assessment, tuple(all_hits)), audit_iterations)

        if last_state is None or last_state.plan is None:
            raise RuntimeError("pipeline did not produce any iterations")
        assert last_assessment is not None
        final_state = self.synthesizer.finalize(
            question,
            last_state.plan,
            tuple(all_hits),
            last_assessment,
            self.max_iterations,
        )
        return _with_audit(_guard_answer(final_state, last_assessment, tuple(all_hits)), audit_iterations)


def _with_audit(answer: GroundedAnswer, audit_iterations: list[dict]) -> GroundedAnswer:
    audit = dict(answer.audit)
    audit["iterations"] = audit_iterations
    return GroundedAnswer(
        answer=answer.answer,
        status=answer.status,
        citations=answer.citations,
        missing_facts=answer.missing_facts,
        sufficiency_score=answer.sufficiency_score,
        conflicts=tuple(answer.conflicts),
        iterations=max(answer.iterations, len(audit_iterations)),
        audit=audit,
    )


def _state_to_audit(state: IterationState) -> dict:
    return {
        "iteration": state.iteration,
        "plan": asdict(state.plan) if state.plan else None,
        "subqueries": [asdict(q) for q in state.subqueries],
        "hits": [
            {
                "id": hit.id,
                "corpus_id": hit.corpus_id,
                "document_id": hit.document_id,
                "score": hit.score,
                "metadata": dict(hit.metadata),
                "text_sha256": sha256(" ".join(hit.text.lower().split()).encode("utf-8")).hexdigest(),
            }
            for hit in state.hits
        ],
        "assessment": asdict(state.assessment) if state.assessment else None,
        "prior_feedback": [asdict(feedback) for feedback in state.prior_feedback],
    }


def _deduplicate_hits(hits: Sequence[Snippet]) -> list[Snippet]:
    """Deduplicate by stable provenance and normalized text, keeping first occurrence."""
    seen: set[str] = set()
    seen_provenance: set[tuple[str, str, str]] = set()
    unique: list[Snippet] = []
    for hit in hits:
        normalized_text = " ".join(hit.text.lower().split())
        provenance_key = (hit.corpus_id, hit.document_id, sha256(normalized_text.encode("utf-8")).hexdigest())
        if hit.id in seen or provenance_key in seen_provenance:
            continue
        seen.add(hit.id)
        seen_provenance.add(provenance_key)
        unique.append(hit)
    return unique


def _validate_corpora(corpora: Sequence[Corpus]) -> None:
    seen: set[str] = set()
    for corpus in corpora:
        if not corpus.id.strip():
            raise ValueError("corpus id must not be empty")
        if corpus.id in seen:
            raise ValueError(f"duplicate corpus id: {corpus.id}")
        seen.add(corpus.id)
        if len(corpus.description.split()) < 3:
            raise ValueError(f"corpus {corpus.id!r} needs a specific routing description")


def _validate_plan(plan, corpora: Sequence[Corpus]) -> None:
    corpus_ids = {corpus.id for corpus in corpora}
    fact_ids = {fact.id for fact in plan.required_facts}
    if not getattr(plan, "required_facts", None):
        raise ValueError("retrieval plan must include required facts")
    if len(fact_ids) != len(plan.required_facts):
        raise ValueError("retrieval plan has duplicate fact ids")
    for route in plan.routes:
        if route.fact_id not in fact_ids:
            raise ValueError(f"route references unknown fact id: {route.fact_id}")
        unknown = set(route.candidate_corpus_ids) - corpus_ids
        if unknown:
            raise ValueError(f"route references unknown corpus ids: {sorted(unknown)}")


def _validate_subqueries(subqueries: Sequence, plan, corpora: Sequence[Corpus]) -> None:
    corpus_ids = {corpus.id for corpus in corpora}
    fact_ids = {fact.id for fact in plan.required_facts}
    for subquery in subqueries:
        if getattr(subquery, "fact_id") not in fact_ids:
            raise ValueError(f"subquery references unknown fact id: {subquery.fact_id}")
        unknown = set(subquery.target_corpus_ids) - corpus_ids
        if unknown:
            raise ValueError(f"subquery references unknown corpus ids: {sorted(unknown)}")
        if not str(getattr(subquery, "query")).strip():
            raise ValueError("subquery query must not be empty")


def _validate_hits(hits: Sequence[Snippet], corpora: Sequence[Corpus]) -> None:
    corpus_ids = {corpus.id for corpus in corpora}
    for hit in hits:
        if hit.corpus_id not in corpus_ids:
            raise ValueError(f"hit references unknown corpus id: {hit.corpus_id}")
        if not hit.id.strip() or not hit.document_id.strip() or not hit.text.strip():
            raise ValueError("retrieval hits must include id, document_id, and text")


def _validate_assessment(
    assessment: ContextAssessment,
    plan,
    hits: Sequence[Snippet],
    corpora: Sequence[Corpus],
) -> None:
    valid_statuses: set[str] = {"sufficient", "insufficient", "irrelevant", "unanswerable"}
    if _enum_value(assessment.status) not in valid_statuses:
        raise ValueError(f"invalid assessment status: {assessment.status}")
    if not 0 <= assessment.sufficiency_score <= 1:
        raise ValueError("assessment sufficiency_score must be between 0 and 1")
    fact_ids = {fact.id for fact in plan.required_facts}
    hit_ids = {hit.id for hit in hits}
    for covered in assessment.covered_facts:
        if covered.fact_id not in fact_ids:
            raise ValueError(f"assessment references unknown fact id: {covered.fact_id}")
        unknown_hits = set(covered.snippet_ids) - hit_ids
        if unknown_hits:
            raise ValueError(f"assessment references unknown snippet ids: {sorted(unknown_hits)}")
    for feedback in assessment.feedback_queries:
        if not getattr(feedback, "query", "").strip():
            raise ValueError("feedback query must not be empty")
        if corpora:
            unknown = set(feedback.target_corpus_ids) - {corpus.id for corpus in corpora}
            if unknown:
                raise ValueError(f"feedback references unknown corpus ids: {sorted(unknown)}")
    if _enum_value(assessment.status) == "sufficient" and assessment.missing_facts:
        raise ValueError("sufficient assessment cannot include missing facts")
    if _enum_value(assessment.status) == "insufficient" and not assessment.missing_facts and not assessment.unsupported_claims:
        raise ValueError("insufficient assessment must explain missing facts or unsupported claims")


def _guard_answer(
    answer: GroundedAnswer,
    assessment: ContextAssessment,
    hits: tuple[Snippet, ...],
) -> GroundedAnswer:
    answer = apply_selective_abstention_policy(answer, assessment)
    known_snippet_ids = {snippet.id for snippet in hits}
    grounded_citations = tuple(
        GroundedCitation(
            claim=citation.claim,
            snippet_ids=tuple(snippet_id for snippet_id in citation.snippet_ids if snippet_id in known_snippet_ids),
        )
        for citation in answer.citations
        if any(snippet_id in known_snippet_ids for snippet_id in citation.snippet_ids)
    )
    if len(grounded_citations) != len(tuple(answer.citations)):
        return GroundedAnswer(
            answer=answer.answer,
            citations=grounded_citations,
            status=AnswerStatus.PARTIAL,
            missing_facts=tuple(answer.missing_facts) or tuple(assessment.missing_facts),
            sufficiency_score=assessment.sufficiency_score,
            conflicts=tuple(answer.conflicts) or tuple(assessment.conflicts),
        )
    return answer
