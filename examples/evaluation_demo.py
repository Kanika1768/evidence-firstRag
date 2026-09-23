import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.in_memory import (
    EvidenceCoverageJudge,
    FeedbackAwareQueryRewriter,
    InMemoryDocument,
    InMemoryRetriever,
    ScriptedPlanner,
    SnippetDrafter,
)
from agentic_rag.contracts import (
    AnswerStatus,
    Corpus,
    GroundedAnswer,
    GroundedCitation,
    ContextStatus,
    RequiredFact,
    RetrievalPlan,
    Route,
)
from agentic_rag.evaluation import EvaluationFixture, ExpectedFetch, compare_runs
from agentic_rag.orchestrator import AgenticRAGOrchestrator, OrchestratorConfig


class ExpectedAnswerSynthesizer:
    def __init__(self, expected_answer: str):
        self.expected_answer = expected_answer

    def synthesize(self, question, plan, snippets, assessment):
        citations = tuple(
            GroundedCitation(
                covered.fact_id,
                tuple(covered.snippet_ids),
            )
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


def build_fixture():
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
        expected_citation_fact_ids=(
            "person_project",
            "project_owner",
        ),
        distractor_corpus_ids=("noise",),
    )


def run_system(fixture, max_iterations):
    plan = RetrievalPlan(
        fixture.question,
        required_facts=(
            RequiredFact(
                "person_project",
                "Alice project",
                metadata={
                    "required_terms": ("alice", "project zen")
                },
            ),
            RequiredFact(
                "project_owner",
                "Project Zen owner",
                metadata={
                    "required_terms": ("project zen", "owner", "nina")
                },
            ),
        ),
        routes=(
            Route(
                "person_project",
                ("directory",),
                "People records contain project assignments.",
            ),
            Route(
                "project_owner",
                ("projects",),
                "Project records contain owners.",
            ),
        ),
    )

    orchestrator = AgenticRAGOrchestrator(
        planner=ScriptedPlanner(plan),
        rewriter=FeedbackAwareQueryRewriter(
            initial_fact_ids=("person_project",)
        ),
        retriever=InMemoryRetriever(
            (
                InMemoryDocument(
                    "directory",
                    "people-1",
                    "Alice works on Project Zen.",
                ),
                InMemoryDocument(
                    "projects",
                    "project-1",
                    "Project Zen owner is Nina.",
                ),
                InMemoryDocument(
                    "noise",
                    "lunch-1",
                    "Nina likes noodles.",
                ),
            )
        ),
        drafter=SnippetDrafter(),
        judge=EvidenceCoverageJudge(),
        synthesizer=ExpectedAnswerSynthesizer(
            "Nina owns Alice's project."
        ),
        config=OrchestratorConfig(
            max_iterations=max_iterations
        ),
    )

    return orchestrator.run(
        fixture.question,
        fixture.corpora,
    )


def print_report(name, report):
    metrics = report.metrics

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Fact coverage:       {metrics.fact_coverage:.2f}")
    print(f"Fetch coverage:      {metrics.fetch_coverage:.2f}")
    print(f"Reasoning correctness: {metrics.reasoning_correctness:.2f}")
    print(f"Citation completeness: {metrics.citation_completeness:.2f}")
    print(f"Iterations:          {metrics.iteration_count}")
    print(f"Status:              {'PASS' if report.passed else 'FAIL'}")

    if report.missing_fact_ids:
        print(
            "Missing facts:       "
            + ", ".join(report.missing_fact_ids)
        )

    if report.distractor_corpus_hits:
        print(
            "Distractor hits:     "
            + ", ".join(report.distractor_corpus_hits)
        )


def main():
    fixture = build_fixture()

    baseline_result = run_system(
        fixture,
        max_iterations=1,
    )

    evidencefirst_result = run_system(
        fixture,
        max_iterations=2,
    )

    comparison = compare_runs(
        fixture,
        baseline_result,
        evidencefirst_result,
    )

    print("=" * 60)
    print("EVIDENCEFIRST RAG - EVALUATION DEMO")
    print("=" * 60)

    print(f"\nQuestion: {fixture.question}")

    print_report(
        "B0 - SINGLE-SHOT BASELINE",
        comparison.baseline,
    )

    print_report(
        "E1 - ITERATIVE EVIDENCEFIRST",
        comparison.candidate,
    )

    print("\nCOMPARISON")
    print("----------")
    print(
        f"Fact coverage gain:       "
        f"{comparison.candidate.metrics.fact_coverage - comparison.baseline.metrics.fact_coverage:+.2f}"
    )
    print(
        f"Fetch coverage gain:      "
        f"{comparison.candidate.metrics.fetch_coverage - comparison.baseline.metrics.fetch_coverage:+.2f}"
    )
    print(
        f"Citation coverage gain:   "
        f"{comparison.candidate.metrics.citation_completeness - comparison.baseline.metrics.citation_completeness:+.2f}"
    )
    print(
        f"Reasoning gain:            "
        f"{comparison.candidate.metrics.reasoning_correctness - comparison.baseline.metrics.reasoning_correctness:+.2f}"
    )
    print(
        "\nImproved:",
        "YES" if comparison.improved else "NO",
    )


if __name__ == "__main__":
    main()
