"""Dependency-free FRAMES-style evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from agentic_rag.contracts import AnswerStatus, Corpus, RequiredFact, RunResult


@dataclass(frozen=True)
class ExpectedFetch:
    fact_id: str
    corpus_id: str
    document_id: str


@dataclass(frozen=True)
class EvaluationFixture:
    question: str
    corpora: Sequence[Corpus]
    required_facts: Sequence[RequiredFact]
    bridge_fact_ids: Sequence[str] = field(default_factory=tuple)
    expected_answer_terms: Sequence[str] = field(default_factory=tuple)
    expected_fetches: Sequence[ExpectedFetch] = field(default_factory=tuple)
    expected_citation_fact_ids: Sequence[str] = field(default_factory=tuple)
    distractor_corpus_ids: Sequence[str] = field(default_factory=tuple)
    expected_missing_facts: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationMetrics:
    fact_coverage: float
    fetch_coverage: float
    reasoning_correctness: float
    citation_completeness: float
    iteration_count: int


@dataclass(frozen=True)
class EvaluationReport:
    fixture: EvaluationFixture
    metrics: EvaluationMetrics
    passed: bool
    covered_fact_ids: Sequence[str] = field(default_factory=tuple)
    missing_fact_ids: Sequence[str] = field(default_factory=tuple)
    fetched_fact_ids: Sequence[str] = field(default_factory=tuple)
    cited_fact_ids: Sequence[str] = field(default_factory=tuple)
    distractor_corpus_hits: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class EvaluationComparison:
    fixture: EvaluationFixture
    baseline: EvaluationReport
    candidate: EvaluationReport
    improved: bool


def evaluate_run(fixture: EvaluationFixture, result: RunResult) -> EvaluationReport:
    required_fact_ids = tuple(fact.id for fact in fixture.required_facts)
    covered_fact_ids = _covered_fact_ids(result)
    fetched_fact_ids = _fetched_fact_ids(fixture, result)
    cited_fact_ids = _cited_fact_ids(fixture, result)
    missing_fact_ids = tuple(fact_id for fact_id in required_fact_ids if fact_id not in covered_fact_ids)
    distractor_hits = tuple(
        corpus_id
        for corpus_id in fixture.distractor_corpus_ids
        if any(snippet.corpus_id == corpus_id for snippet in result.snippets)
    )

    metrics = EvaluationMetrics(
        fact_coverage=_coverage(covered_fact_ids, required_fact_ids),
        fetch_coverage=_coverage(fetched_fact_ids, tuple(fetch.fact_id for fetch in fixture.expected_fetches)),
        reasoning_correctness=_reasoning_correctness(fixture, result),
        citation_completeness=_coverage(cited_fact_ids, tuple(fixture.expected_citation_fact_ids)),
        iteration_count=len(tuple(result.iterations)),
    )
    passed = (
        metrics.fact_coverage == 1.0
        and metrics.fetch_coverage == 1.0
        and metrics.reasoning_correctness == 1.0
        and metrics.citation_completeness == 1.0
        and not distractor_hits
    )
    return EvaluationReport(
        fixture=fixture,
        metrics=metrics,
        passed=passed,
        covered_fact_ids=covered_fact_ids,
        missing_fact_ids=missing_fact_ids,
        fetched_fact_ids=fetched_fact_ids,
        cited_fact_ids=cited_fact_ids,
        distractor_corpus_hits=distractor_hits,
    )


def compare_runs(
    fixture: EvaluationFixture,
    baseline_result: RunResult,
    candidate_result: RunResult,
) -> EvaluationComparison:
    baseline = evaluate_run(fixture, baseline_result)
    candidate = evaluate_run(fixture, candidate_result)
    return EvaluationComparison(
        fixture=fixture,
        baseline=baseline,
        candidate=candidate,
        improved=_score(candidate) > _score(baseline),
    )


def _covered_fact_ids(result: RunResult) -> tuple[str, ...]:
    if not result.iterations:
        return ()
    return tuple(covered.fact_id for covered in result.iterations[-1].assessment.covered_facts)


def _fetched_fact_ids(fixture: EvaluationFixture, result: RunResult) -> tuple[str, ...]:
    fetched: list[str] = []
    for expected in fixture.expected_fetches:
        if any(
            snippet.corpus_id == expected.corpus_id and snippet.document_id == expected.document_id
            for snippet in result.snippets
        ):
            fetched.append(expected.fact_id)
    return tuple(fetched)


def _cited_fact_ids(fixture: EvaluationFixture, result: RunResult) -> tuple[str, ...]:
    cited: list[str] = []
    for fact_id in fixture.expected_citation_fact_ids:
        if any(citation.claim == fact_id and citation.snippet_ids for citation in result.answer.citations):
            cited.append(fact_id)
    return tuple(cited)


def _reasoning_correctness(fixture: EvaluationFixture, result: RunResult) -> float:
    if _enum_value(result.answer.status) != AnswerStatus.ANSWERED.value:
        return 0.0
    answer_text = result.answer.answer.lower()
    if all(term.lower() in answer_text for term in fixture.expected_answer_terms):
        return 1.0
    return 0.0


def _coverage(found_ids: Sequence[str], expected_ids: Sequence[str]) -> float:
    expected = tuple(dict.fromkeys(expected_ids))
    if not expected:
        return 1.0
    found = set(found_ids)
    return len(tuple(item for item in expected if item in found)) / len(expected)


def _score(report: EvaluationReport) -> float:
    return (
        report.metrics.fact_coverage
        + report.metrics.fetch_coverage
        + report.metrics.reasoning_correctness
        + report.metrics.citation_completeness
    )


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)
