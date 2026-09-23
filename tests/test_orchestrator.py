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
    RuleBasedSynthesizer,
    ScriptedPlanner,
    SnippetDrafter,
)
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge  # noqa: E402
from agentic_rag.contracts import (  # noqa: E402
    AnswerabilityLabel,
    AnswerStatus,
    Claim,
    ContextAssessment,
    ContextStatus,
    Corpus,
    CoveredFact,
    DraftAnswer,
    GroundedAnswer,
    GroundedCitation,
    RequiredFact,
    RetrievalPlan,
    Route,
)
from agentic_rag.orchestrator import AgenticRAGOrchestrator, OrchestratorConfig  # noqa: E402


class AgenticRAGOrchestratorTests(unittest.TestCase):
    def build_two_hop_plan(self):
        return RetrievalPlan(
            question="Who owns Alice's project?",
            required_facts=(
                RequiredFact(
                    id="person_project",
                    description="Alice's project name",
                    metadata={"required_terms": ("alice", "project zen")},
                ),
                RequiredFact(
                    id="project_owner",
                    description="Project Zen owner",
                    metadata={"required_terms": ("project zen", "owner", "nina")},
                ),
            ),
            routes=(
                Route("person_project", ("directory",), "People records contain project assignments."),
                Route("project_owner", ("projects",), "Project records contain owners."),
            ),
            stop_conditions=("Stop when both required facts are covered.",),
        )

    def test_iterates_on_missing_fact_without_searching_every_corpus(self):
        corpora = (
            Corpus("directory", "Employee directory and project assignments."),
            Corpus("projects", "Project ownership records."),
            Corpus("noise", "Unrelated cafeteria notices."),
        )
        retriever = InMemoryRetriever(
            (
                InMemoryDocument("directory", "people-1", "Alice works on Project Zen."),
                InMemoryDocument("projects", "project-1", "Project Zen owner is Nina."),
                InMemoryDocument("noise", "lunch-1", "Nina likes noodles."),
            )
        )
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(self.build_two_hop_plan()),
            rewriter=FeedbackAwareQueryRewriter(initial_fact_ids=("person_project",)),
            retriever=retriever,
            drafter=SnippetDrafter(),
            judge=EvidenceCoverageJudge(),
            synthesizer=RuleBasedSynthesizer(),
            config=OrchestratorConfig(max_iterations=2),
        )

        result = orchestrator.run("Who owns Alice's project?", corpora)

        self.assertEqual(ContextStatus.SUFFICIENT, result.iterations[-1].assessment.status)
        self.assertEqual(AnswerStatus.ANSWERED, result.answer.status)
        self.assertEqual(2, len(result.iterations))
        self.assertEqual(1.0, result.answer.sufficiency_score)
        self.assertTrue(result.answer.citations)
        self.assertTrue(all(citation.snippet_ids for citation in result.answer.citations))
        self.assertEqual(("directory",), retriever.calls[0].target_corpus_ids)
        self.assertEqual(("projects",), retriever.calls[1].target_corpus_ids)
        self.assertNotIn(("directory", "projects", "noise"), [call.target_corpus_ids for call in retriever.calls])

    def test_missing_evidence_returns_unanswerable_with_feedback_query(self):
        plan = RetrievalPlan(
            question="Who approved Project Atlas?",
            required_facts=(
                RequiredFact(
                    id="approval",
                    description="Project Atlas approver",
                    metadata={"required_terms": ("project atlas", "approved")},
                ),
            ),
            routes=(Route("approval", ("approvals",), "Approval records contain approvers."),),
            stop_conditions=("Stop after approval evidence is found.",),
        )
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(plan),
            rewriter=FeedbackAwareQueryRewriter(),
            retriever=InMemoryRetriever(()),
            drafter=SnippetDrafter(),
            judge=EvidenceCoverageJudge(),
            synthesizer=RuleBasedSynthesizer(),
            config=OrchestratorConfig(max_iterations=1),
        )

        result = orchestrator.run("Who approved Project Atlas?", (Corpus("approvals", "Approval records."),))

        self.assertEqual(AnswerStatus.UNANSWERABLE, result.answer.status)
        self.assertIn("Project Atlas approver", result.answer.missing_facts)
        self.assertEqual(0.0, result.answer.sufficiency_score)
        self.assertEqual("Project Atlas approver", result.iterations[-1].assessment.feedback_queries[0].query)

    def test_unsupported_draft_claim_fails_sufficiency(self):
        plan = self.build_two_hop_plan()
        judge = EvidenceCoverageJudge()
        assessment = judge.assess(
            question=plan.question,
            plan=plan,
            snippets=(),
            draft=DraftAnswer(claims=(Claim("Alice's project is Project Zen.", ()),)),
        )

        self.assertEqual(ContextStatus.INSUFFICIENT, assessment.status)
        self.assertIn("Alice's project is Project Zen.", assessment.unsupported_claims)
        self.assertLess(assessment.sufficiency_score, 1.0)

    def test_stops_after_empty_subquery_iteration(self):
        plan = self.build_two_hop_plan()
        retriever = InMemoryRetriever(())
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(plan),
            rewriter=FeedbackAwareQueryRewriter(initial_fact_ids=()),
            retriever=retriever,
            drafter=SnippetDrafter(),
            judge=EvidenceCoverageJudge(),
            synthesizer=RuleBasedSynthesizer(),
            config=OrchestratorConfig(max_iterations=3),
        )

        result = orchestrator.run(plan.question, (Corpus("directory", "People records."),))

        self.assertEqual(1, len(result.iterations))
        self.assertEqual((), result.iterations[0].subqueries)
        self.assertEqual([], retriever.calls)

    def test_downgrades_answer_with_unknown_citation_snippet(self):
        class UnknownCitationSynthesizer:
            def synthesize(self, question, plan, snippets, assessment):
                return GroundedAnswer(
                    answer="Unsupported citation.",
                    citations=(GroundedCitation("project_owner", ("missing-snippet",)),),
                    status=AnswerStatus.ANSWERED,
                    sufficiency_score=assessment.sufficiency_score,
                )

        corpora = (
            Corpus("directory", "Employee directory and project assignments."),
            Corpus("projects", "Project ownership records."),
        )
        retriever = InMemoryRetriever(
            (
                InMemoryDocument("directory", "people-1", "Alice works on Project Zen."),
                InMemoryDocument("projects", "project-1", "Project Zen owner is Nina."),
            )
        )
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(self.build_two_hop_plan()),
            rewriter=FeedbackAwareQueryRewriter(),
            retriever=retriever,
            drafter=SnippetDrafter(),
            judge=EvidenceCoverageJudge(),
            synthesizer=UnknownCitationSynthesizer(),
            config=OrchestratorConfig(max_iterations=1),
        )

        result = orchestrator.run("Who owns Alice's project?", corpora)

        self.assertEqual(AnswerStatus.PARTIAL, result.answer.status)
        self.assertEqual((), result.answer.citations)

    def test_answerability_policy_downgrades_useful_incomplete_answer(self):
        class UsefulIncompleteJudge:
            def assess(self, question, plan, snippets, draft):
                return ContextAssessment(
                    status=ContextStatus.INSUFFICIENT,
                    sufficiency_score=0.5,
                    covered_facts=(CoveredFact("person_project", tuple(snippet.id for snippet in snippets)),),
                    missing_facts=("Project Zen owner",),
                    answerability=AnswerabilityLabel.USEFUL_BUT_INCOMPLETE,
                )

        class OvereagerSynthesizer:
            def synthesize(self, question, plan, snippets, assessment):
                return GroundedAnswer(
                    answer="Alice works on Project Zen, so Nina owns it.",
                    citations=(GroundedCitation("person_project", (snippets[0].id,)),),
                    status=AnswerStatus.ANSWERED,
                    sufficiency_score=assessment.sufficiency_score,
                )

        plan = self.build_two_hop_plan()
        retriever = InMemoryRetriever((InMemoryDocument("directory", "people-1", "Alice works on Project Zen."),))
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(plan),
            rewriter=FeedbackAwareQueryRewriter(initial_fact_ids=("person_project",)),
            retriever=retriever,
            drafter=SnippetDrafter(),
            judge=UsefulIncompleteJudge(),
            synthesizer=OvereagerSynthesizer(),
            config=OrchestratorConfig(max_iterations=1),
        )

        result = orchestrator.run(plan.question, (Corpus("directory", "People records."),))

        self.assertEqual(AnswerStatus.PARTIAL, result.answer.status)
        self.assertEqual(("Project Zen owner",), tuple(result.answer.missing_facts))
        self.assertIn("Partial evidence", result.answer.answer)
        self.assertTrue(result.answer.citations)

    def test_conflicting_evidence_is_surfaced_with_both_sides_cited(self):
        plan = RetrievalPlan(
            question="Who owns Project Zen?",
            required_facts=(
                RequiredFact(
                    id="owner",
                    description="Project Zen owner",
                    metadata={
                        "required_terms": ("project zen", "owner"),
                        "conflict_terms": ("nina", "omar"),
                    },
                ),
            ),
            routes=(Route("owner", ("projects",), "Project records contain owners."),),
        )
        retriever = InMemoryRetriever(
            (
                InMemoryDocument("projects", "project-1", "Project Zen owner is Nina."),
                InMemoryDocument("projects", "project-2", "Project Zen owner is Omar."),
            )
        )
        orchestrator = AgenticRAGOrchestrator(
            planner=ScriptedPlanner(plan),
            rewriter=FeedbackAwareQueryRewriter(initial_fact_ids=("owner",)),
            retriever=retriever,
            drafter=SnippetDrafter(),
            judge=AutoraterStyleSufficiencyJudge(),
            synthesizer=RuleBasedSynthesizer(),
            config=OrchestratorConfig(max_iterations=1),
        )

        result = orchestrator.run(plan.question, (Corpus("projects", "Project ownership records."),))

        self.assertEqual(AnswerStatus.PARTIAL, result.answer.status)
        self.assertEqual(AnswerabilityLabel.CONFLICTING, result.iterations[-1].assessment.answerability_label)
        self.assertEqual("owner", result.answer.conflicts[0].fact_id)
        self.assertEqual(2, len(result.answer.conflicts[0].groups))
        self.assertEqual(("nina", "omar"), tuple(group.label for group in result.answer.conflicts[0].groups))
        self.assertEqual(("owner:nina", "owner:omar"), tuple(citation.claim for citation in result.answer.citations))
        self.assertTrue(all(citation.snippet_ids for citation in result.answer.citations))
        self.assertIn("Conflicting evidence", result.answer.answer)


if __name__ == "__main__":
    unittest.main()
