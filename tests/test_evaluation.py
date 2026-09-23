import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.in_memory import (  # noqa: E402
    EvidenceCoverageJudge,
    FeedbackAwareQueryRewriter,
    InMemoryDocument,
    InMemoryRetriever,
    ScriptedPlanner,
    SnippetDrafter,
)
from agentic_rag.contracts import (  # noqa: E402
    AnswerStatus,
    ContextAssessment,
    ContextStatus,
    Corpus,
    CoveredFact,
    DraftAnswer,
    GroundedAnswer,
    GroundedCitation,
    IterationTrace,
    RequiredFact,
    RetrievalPlan,
    Route,
    RunResult,
    Snippet,
)
from agentic_rag.evaluation import EvaluationFixture, ExpectedFetch, compare_runs, evaluate_run  # noqa: E402
from agentic_rag.orchestrator import AgenticRAGOrchestrator, OrchestratorConfig  # noqa: E402


class FramesStyleEvaluationTests(unittest.TestCase):
    def test_fixture_records_bridge_facts_expected_fetches_and_distractors(self):
        fixture = _fixture()

        self.assertEqual("Who owns Alice's project?", fixture.question)
        self.assertEqual(("person_project",), fixture.bridge_fact_ids)
        self.assertEqual(("noise",), fixture.distractor_corpus_ids)
        self.assertEqual(
            (
                ExpectedFetch("person_project", "directory", "people-1"),
                ExpectedFetch("project_owner", "projects", "project-1"),
            ),
            fixture.expected_fetches,
        )

    def test_evaluate_run_reports_full_fact_fetch_reasoning_and_citation_scores(self):
        fixture = _fixture()
        result = _run_result(
            fixture,
            snippets=(
                Snippet("s1", "directory", "people-1", "Alice works on Project Zen.", fact_id="person_project"),
                Snippet("s2", "projects", "project-1", "Project Zen owner is Nina.", fact_id="project_owner"),
            ),
            covered_fact_ids=("person_project", "project_owner"),
            answer=GroundedAnswer(
                answer="Nina owns Alice's project.",
                citations=(
                    GroundedCitation("person_project", ("s1",)),
                    GroundedCitation("project_owner", ("s2",)),
                ),
                status=AnswerStatus.ANSWERED,
                sufficiency_score=1.0,
            ),
            iteration_count=2,
        )

        report = evaluate_run(fixture, result)

        self.assertEqual(1.0, report.metrics.fact_coverage)
        self.assertEqual(1.0, report.metrics.fetch_coverage)
        self.assertEqual(1.0, report.metrics.reasoning_correctness)
        self.assertEqual(1.0, report.metrics.citation_completeness)
        self.assertEqual(2, report.metrics.iteration_count)
        self.assertTrue(report.passed)
        self.assertEqual((), report.distractor_corpus_hits)

    def test_evaluate_run_reports_partial_scores_for_missing_second_hop(self):
        fixture = _fixture()
        result = _run_result(
            fixture,
            snippets=(Snippet("s1", "directory", "people-1", "Alice works on Project Zen.", fact_id="person_project"),),
            covered_fact_ids=("person_project",),
            answer=GroundedAnswer(
                answer="Partial evidence found.",
                citations=(GroundedCitation("person_project", ("s1",)),),
                status=AnswerStatus.PARTIAL,
                missing_facts=("Project Zen owner",),
                sufficiency_score=0.5,
            ),
            iteration_count=1,
        )

        report = evaluate_run(fixture, result)

        self.assertEqual(0.5, report.metrics.fact_coverage)
        self.assertEqual(0.5, report.metrics.fetch_coverage)
        self.assertEqual(0.0, report.metrics.reasoning_correctness)
        self.assertEqual(0.5, report.metrics.citation_completeness)
        self.assertEqual(1, report.metrics.iteration_count)
        self.assertFalse(report.passed)
        self.assertEqual(("project_owner",), report.missing_fact_ids)

    def test_evaluate_run_reports_distractor_corpus_hits(self):
        fixture = _fixture()
        result = _run_result(
            fixture,
            snippets=(
                Snippet("s1", "directory", "people-1", "Alice works on Project Zen.", fact_id="person_project"),
                Snippet("s3", "noise", "lunch-1", "Nina likes noodles.", fact_id="project_owner"),
            ),
            covered_fact_ids=("person_project",),
            answer=GroundedAnswer(
                answer="Partial evidence found.",
                citations=(GroundedCitation("person_project", ("s1",)),),
                status=AnswerStatus.PARTIAL,
                missing_facts=("Project Zen owner",),
                sufficiency_score=0.5,
            ),
            iteration_count=1,
        )

        report = evaluate_run(fixture, result)

        self.assertEqual(("noise",), report.distractor_corpus_hits)
        self.assertFalse(report.passed)

    def test_iterative_retrieval_passes_where_single_shot_fails(self):
        fixture = _fixture()
        single_shot = _run_orchestrator(fixture, max_iterations=1)
        iterative = _run_orchestrator(fixture, max_iterations=2)

        comparison = compare_runs(fixture, single_shot, iterative)

        self.assertFalse(comparison.baseline.passed)
        self.assertTrue(comparison.candidate.passed)
        self.assertTrue(comparison.improved)
        self.assertEqual(1, comparison.baseline.metrics.iteration_count)
        self.assertEqual(2, comparison.candidate.metrics.iteration_count)
        self.assertEqual(0.5, comparison.baseline.metrics.fetch_coverage)
        self.assertEqual(1.0, comparison.candidate.metrics.fetch_coverage)
        self.assertEqual(0.5, comparison.baseline.metrics.citation_completeness)
        self.assertEqual(1.0, comparison.candidate.metrics.citation_completeness)
        self.assertEqual(("project_owner",), comparison.baseline.missing_fact_ids)
        self.assertEqual((), comparison.candidate.missing_fact_ids)


def _fixture() -> EvaluationFixture:
    return EvaluationFixture(
        question="Who owns Alice's project?",
        corpora=(
            Corpus("directory", "Employee directory and project assignments."),
            Corpus("projects", "Project ownership records."),
            Corpus("noise", "Unrelated cafeteria notices."),
        ),
        required_facts=(
            RequiredFact("person_project", "Alice project"),
            RequiredFact("project_owner", "Project Zen owner"),
        ),
        bridge_fact_ids=("person_project",),
        expected_answer_terms=("nina",),
        expected_fetches=(
            ExpectedFetch("person_project", "directory", "people-1"),
            ExpectedFetch("project_owner", "projects", "project-1"),
        ),
        expected_citation_fact_ids=("person_project", "project_owner"),
        distractor_corpus_ids=("noise",),
    )


def _run_result(
    fixture: EvaluationFixture,
    *,
    snippets: tuple[Snippet, ...],
    covered_fact_ids: tuple[str, ...],
    answer: GroundedAnswer,
    iteration_count: int,
) -> RunResult:
    plan = RetrievalPlan(fixture.question, fixture.required_facts, routes=())
    assessment = ContextAssessment(
        status=ContextStatus.SUFFICIENT if answer.status == AnswerStatus.ANSWERED else ContextStatus.INSUFFICIENT,
        sufficiency_score=answer.sufficiency_score,
        covered_facts=tuple(CoveredFact(fact_id, tuple(snippet.id for snippet in snippets if snippet.fact_id == fact_id)) for fact_id in covered_fact_ids),
        missing_facts=tuple(answer.missing_facts),
    )
    traces = tuple(
        IterationTrace(
            iteration=index,
            subqueries=(),
            snippets=snippets if index == iteration_count - 1 else (),
            draft=DraftAnswer(),
            assessment=assessment,
        )
        for index in range(iteration_count)
    )
    return RunResult(
        question=fixture.question,
        plan=plan,
        answer=answer,
        iterations=traces,
        snippets=snippets,
    )


def _run_orchestrator(fixture: EvaluationFixture, *, max_iterations: int) -> RunResult:
    plan = RetrievalPlan(
        fixture.question,
        required_facts=(
            RequiredFact(
                "person_project",
                "Alice project",
                metadata={"required_terms": ("alice", "project zen")},
            ),
            RequiredFact(
                "project_owner",
                "Project Zen owner",
                metadata={"required_terms": ("project zen", "owner", "nina")},
            ),
        ),
        routes=(
            Route("person_project", ("directory",), "People records contain project assignments."),
            Route("project_owner", ("projects",), "Project records contain owners."),
        ),
    )
    orchestrator = AgenticRAGOrchestrator(
        planner=ScriptedPlanner(plan),
        rewriter=FeedbackAwareQueryRewriter(initial_fact_ids=("person_project",)),
        retriever=InMemoryRetriever(
            (
                InMemoryDocument("directory", "people-1", "Alice works on Project Zen."),
                InMemoryDocument("projects", "project-1", "Project Zen owner is Nina."),
                InMemoryDocument("noise", "lunch-1", "Nina likes noodles."),
            )
        ),
        drafter=SnippetDrafter(),
        judge=EvidenceCoverageJudge(),
        synthesizer=ExpectedAnswerSynthesizer("Nina owns Alice's project."),
        config=OrchestratorConfig(max_iterations=max_iterations),
    )
    return orchestrator.run(fixture.question, fixture.corpora)


class ExpectedAnswerSynthesizer:
    def __init__(self, expected_answer: str) -> None:
        self.expected_answer = expected_answer

    def synthesize(self, question, plan, snippets, assessment):
        citations = tuple(
            GroundedCitation(covered.fact_id, tuple(covered.snippet_ids))
            for covered in assessment.covered_facts
        )
        if assessment.status == ContextStatus.SUFFICIENT:
            return GroundedAnswer(
                answer=self.expected_answer,
                citations=citations,
                status=AnswerStatus.ANSWERED,
                sufficiency_score=assessment.sufficiency_score,
            )
        return GroundedAnswer(
            answer="Partial evidence found.",
            citations=citations,
            status=AnswerStatus.PARTIAL if citations else AnswerStatus.UNANSWERABLE,
            missing_facts=tuple(assessment.missing_facts),
            sufficiency_score=assessment.sufficiency_score,
        )


if __name__ == "__main__":
    unittest.main()
