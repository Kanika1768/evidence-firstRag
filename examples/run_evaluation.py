"""Reproducible evaluation harness comparing B0 Baseline, B1 Sufficiency-Aware RAG, and E1 EvidenceFirst RAG."""

from __future__ import annotations

import csv
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.adapters.retriever import LexicalRetriever
from agentic_rag.contracts import (
    AnswerStatus,
    AnswerabilityLabel,
    ContextAssessment,
    ContextStatus,
    Corpus,
    GroundedAnswer,
    GroundedCitation,
    IterationTrace,
    RetrievalPlan,
    Route,
    RunResult,
    Snippet,
    Subquery,
)
from agentic_rag.document_index import build_local_index
from agentic_rag.document_loader import load_documents
from agentic_rag.general_planner import GeneralPlanner
from agentic_rag.generic_pipeline import (
    _best_evidence_sentence,
    build_generic_components,
)
from agentic_rag.orchestrator import (
    AgenticRAGOrchestrator,
    OrchestratorConfig,
)
from agentic_rag.sufficiency import (
    AutoraterStyleSufficiencyJudge,
    apply_selective_abstention_policy,
)

DATA_DIR = Path("data/demo_documents")
RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class EvaluationTestCase:
    id: str
    question: str
    expected_label: AnswerabilityLabel
    expected_terms: Sequence[str] = field(default_factory=tuple)
    target_corpora: Sequence[str] = field(default_factory=tuple)
    initial_corpus_mask: Sequence[str] = field(default_factory=tuple)
    description: str = ""


@dataclass(frozen=True)
class EvaluationRunRecord:
    system_name: str
    test_case: EvaluationTestCase
    answer: GroundedAnswer
    iterations: int
    snippets: Sequence[Snippet]
    is_correct: bool
    is_grounded: bool
    is_unsupported: bool
    is_abstained: bool
    matched_expected: bool
    latency_ms: float = 0.0
    diagnostic: str = ""


@dataclass(frozen=True)
class EvaluationAggregate:
    system_name: str
    total_questions: int
    correct_count: int
    grounded_count: int
    unsupported_count: int
    abstained_count: int
    total_iterations: int
    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    recovery_gain: int | None = None

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.total_questions if self.total_questions else 0.0

    @property
    def groundedness_rate(self) -> float:
        return self.grounded_count / self.total_questions if self.total_questions else 0.0

    @property
    def unsupported_rate(self) -> float:
        return self.unsupported_count / self.total_questions if self.total_questions else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.abstained_count / self.total_questions if self.total_questions else 0.0

    @property
    def average_iterations(self) -> float:
        return self.total_iterations / self.total_questions if self.total_questions else 0.0


# ---------------------------------------------------------------------------
# Benchmark Test Set (NovaTech Corpora)
# ---------------------------------------------------------------------------

BENCHMARK_TEST_CASES: tuple[EvaluationTestCase, ...] = (
    EvaluationTestCase(
        id="Q1",
        question="Who founded NovaTech and when was it founded?",
        expected_label=AnswerabilityLabel.SUFFICIENT,
        expected_terms=("priya mehta", "2019"),
        target_corpora=("company", "history"),
        description="Multi-corpus fact requirement: founder in company.txt, founding year in history.txt.",
    ),
    EvaluationTestCase(
        id="Q2",
        question="What does NovaTech develop?",
        expected_label=AnswerabilityLabel.SUFFICIENT,
        expected_terms=("analytics software",),
        target_corpora=("company",),
        description="Single-corpus factual question answerable from company.txt.",
    ),
    EvaluationTestCase(
        id="Q3",
        question="What is NovaTech's annual revenue?",
        expected_label=AnswerabilityLabel.SUFFICIENT,
        expected_terms=("42 million",),
        target_corpora=("finance",),
        description="Single-corpus financial question answerable from finance.txt.",
    ),
    EvaluationTestCase(
        id="Q4",
        question="Who is the CEO of NovaTech?",
        expected_label=AnswerabilityLabel.UNANSWERABLE,
        expected_terms=(),
        target_corpora=(),
        description="Unanswerable question: CEO information is absent from all documents.",
    ),
    EvaluationTestCase(
        id="Q5",
        question="What is NovaTech's stock price?",
        expected_label=AnswerabilityLabel.UNANSWERABLE,
        expected_terms=(),
        target_corpora=(),
        description="Unanswerable question: stock price information is absent from all documents.",
    ),
    EvaluationTestCase(
        id="Q6",
        question="How many employees does NovaTech have?",
        expected_label=AnswerabilityLabel.UNANSWERABLE,
        expected_terms=(),
        target_corpora=(),
        description="Unanswerable question: employee count is absent from all documents.",
    ),
    EvaluationTestCase(
        id="Q7",
        question="What is NovaTech's annual revenue?",
        expected_label=AnswerabilityLabel.SUFFICIENT,
        expected_terms=("42 million",),
        target_corpora=("finance",),
        initial_corpus_mask=("finance",),
        description="Controlled recovery scenario: finance corpus masked on pass 1, requiring iterative feedback recovery.",
    ),
)


# ---------------------------------------------------------------------------
# B0 Baseline Implementation (Conventional One-Shot RAG)
# ---------------------------------------------------------------------------

class B0ConventionalRAG:
    """Conventional one-shot RAG baseline.

    Characteristics:
    - Single retrieval call across available corpora using the raw question.
    - Directly extracts best matching sentence from top retrieved hit.
    - No sufficiency judge to evaluate fact-level support.
    - No selective abstention policy (always answers, never abstains).
    - No recovery iterations (strictly one shot).
    """

    def __init__(self, retriever: LexicalRetriever) -> None:
        self.retriever = retriever

    def run(
        self,
        question: str,
        corpora: Sequence[Corpus],
        mask: Sequence[str] = (),
    ) -> RunResult:
        searchable_corpora = [c for c in corpora if c.id not in mask] or list(corpora)
        subquery = Subquery(
            id="b0-oneshot",
            fact_id="b0-query",
            query=question,
            target_corpus_ids=tuple(c.id for c in searchable_corpora),
            reason="One-shot retrieval across available corpora.",
            iteration=0,
        )
        hits = tuple(self.retriever.retrieve(subquery))

        if hits:
            extracted_claim = _best_evidence_sentence(question, hits[0].text)
            answer = GroundedAnswer(
                answer=extracted_claim,
                citations=(GroundedCitation(claim=extracted_claim, snippet_ids=(hits[0].id,)),),
                status=AnswerStatus.ANSWERED,
                sufficiency_score=1.0,
                iterations=1,
            )
        else:
            answer = GroundedAnswer(
                answer="No relevant documents retrieved.",
                citations=(),
                status=AnswerStatus.ANSWERED,
                sufficiency_score=0.0,
                iterations=1,
            )

        dummy_plan = RetrievalPlan(
            question=question,
            required_facts=(),
            routes=(Route("b0-fact", tuple(c.id for c in searchable_corpora), "One-shot route"),),
        )
        dummy_trace = IterationTrace(
            iteration=0,
            subqueries=(subquery,),
            snippets=hits,
            draft=None,  # type: ignore[arg-type]
            assessment=ContextAssessment(status=ContextStatus.SUFFICIENT, sufficiency_score=1.0),
        )

        return RunResult(
            question=question,
            plan=dummy_plan,
            answer=answer,
            iterations=(dummy_trace,),
            snippets=hits,
        )


# ---------------------------------------------------------------------------
# B1 Baseline Implementation (Sufficiency-Aware One-Shot RAG)
# ---------------------------------------------------------------------------

class B1SufficiencyAwareRAG:
    """Sufficiency-Aware One-Shot RAG baseline (B1).

    Characteristics:
    - Guided one-shot retrieval using planner subqueries routed across corpora.
    - Evaluates retrieved context using AutoraterStyleSufficiencyJudge under Diligent Reader rules.
    - If context is SUFFICIENT: synthesizes grounded answer with citations.
    - If context is INSUFFICIENT or UNANSWERABLE: immediately applies selective abstention policy.
    - Strictly ZERO recovery iterations (single retrieval attempt, iterations=1).
    """

    def __init__(
        self,
        retriever: LexicalRetriever,
        planner: GeneralPlanner,
        rewriter: Any,
        judge: Any,
        drafter: Any,
        synthesizer: Any,
    ) -> None:
        self.retriever = retriever
        self.planner = planner
        self.rewriter = rewriter
        self.judge = judge
        self.drafter = drafter
        self.synthesizer = synthesizer

    def run(
        self,
        question: str,
        corpora: Sequence[Corpus],
        mask: Sequence[str] = (),
    ) -> RunResult:
        plan = self.planner.plan(question, tuple(corpora), prior_assessment=None)
        subqueries = tuple(self.rewriter.rewrite(question, plan, prior_assessment=None, iteration=0))

        if mask:
            masked_subqueries = []
            for sq in subqueries:
                remaining_targets = tuple(c for c in sq.target_corpus_ids if c not in mask)
                target = remaining_targets or ("company", "history")
                masked_subqueries.append(
                    Subquery(
                        id=sq.id,
                        fact_id=sq.fact_id,
                        query=sq.query,
                        target_corpus_ids=target,
                        reason=f"Initial search excluding masked corpora {list(mask)}",
                        iteration=0,
                    )
                )
            subqueries = tuple(masked_subqueries)

        snippets_by_id: dict[str, Snippet] = {}
        retrieved: list[Snippet] = []
        for sq in subqueries:
            for snip in self.retriever.retrieve(sq):
                if snip.id not in snippets_by_id:
                    snippets_by_id[snip.id] = snip
                    retrieved.append(snip)

        snippets_tuple = tuple(snippets_by_id.values())
        draft = self.drafter.draft(question, plan, snippets_tuple)
        assessment = self.judge.assess(question, plan, snippets_tuple, draft)
        candidate_answer = self.synthesizer.synthesize(question, plan, snippets_tuple, assessment)
        final_answer = apply_selective_abstention_policy(candidate_answer, assessment)

        trace = IterationTrace(
            iteration=0,
            subqueries=subqueries,
            snippets=tuple(retrieved),
            draft=draft,
            assessment=assessment,
        )

        return RunResult(
            question=question,
            plan=plan,
            answer=final_answer,
            iterations=(trace,),
            snippets=snippets_tuple,
        )


# ---------------------------------------------------------------------------
# Controlled Recovery Rewriter Demonstration Component
# ---------------------------------------------------------------------------

class ControlledRecoveryRewriter:
    """Rewriter supporting controlled masking on pass 1 and targeted recovery on pass 2."""

    def __init__(self, base_rewriter: Any, default_mask_revenue: bool = True) -> None:
        self.base_rewriter = base_rewriter
        self.default_mask_revenue = default_mask_revenue
        self.active_mask: Sequence[str] = ()

    def set_mask(self, mask: Sequence[str]) -> None:
        self.active_mask = tuple(mask)

    def rewrite(
        self,
        question: str,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> tuple[Subquery, ...]:
        queries = tuple(
            self.base_rewriter.rewrite(question, plan, prior_assessment, iteration)
        )
        should_mask = bool(self.active_mask) or (
            self.default_mask_revenue and question.strip() == "What is NovaTech's annual revenue?"
        )
        target_mask = tuple(self.active_mask) or ("finance",)
        if should_mask:
            if prior_assessment is None:
                # Iteration 0: Mask designated corpora to simulate initial routing omission
                return tuple(
                    Subquery(
                        id=q.id,
                        fact_id=q.fact_id,
                        query=q.query,
                        target_corpus_ids=tuple(c for c in q.target_corpus_ids if c not in target_mask) or ("company", "history"),
                        reason=f"Iteration 0: Initial retrieval omitting {list(target_mask)} (controlled masking).",
                        iteration=iteration,
                    )
                    for q in queries
                )
            else:
                # Iteration 1: Targeted recovery query to the previously missing corpora
                return tuple(
                    Subquery(
                        id=q.id,
                        fact_id=q.fact_id,
                        query=q.query,
                        target_corpus_ids=tuple(target_mask),
                        reason=f"Iteration 1: Targeted recovery query to recovered corpora {list(target_mask)}.",
                        iteration=iteration,
                    )
                    for q in queries
                )
        return queries


# ---------------------------------------------------------------------------
# Evaluation & Scoring Logic
# ---------------------------------------------------------------------------

def evaluate_run_record(
    test_case: EvaluationTestCase,
    result: RunResult,
    system_name: str,
    latency_ms: float = 0.0,
) -> EvaluationRunRecord:
    answer = result.answer
    status_str = answer.status.value if hasattr(answer.status, "value") else str(answer.status)
    answer_text_lower = answer.answer.lower()
    iterations = len(result.iterations)

    is_abstained = status_str in (AnswerStatus.UNANSWERABLE.value, AnswerStatus.PARTIAL.value)
    is_answered = status_str == AnswerStatus.ANSWERED.value

    if test_case.expected_label == AnswerabilityLabel.SUFFICIENT:
        has_all_terms = all(term.lower() in answer_text_lower for term in test_case.expected_terms)
        is_correct = is_answered and has_all_terms
        is_grounded = bool(answer.citations) and is_correct
        is_unsupported = False
        matched_expected = is_correct

        if is_correct:
            diagnostic = "MATCH (All expected facts retrieved, verified, and grounded)."
        elif is_answered and not has_all_terms:
            diagnostic = "PARTIAL / MISSED FACT (Answered, but missed one or more required facts)."
        else:
            diagnostic = "UNEXPECTED ABSTENTION (Failed to retrieve sufficient available evidence)."

    else:  # UNANSWERABLE
        is_correct = is_abstained
        is_grounded = is_abstained
        is_unsupported = is_answered
        matched_expected = is_abstained

        if is_abstained:
            diagnostic = "MATCH (Correctly detected missing evidence and selectively abstained)."
        else:
            diagnostic = "UNSUPPORTED ANSWER (Answered with unverified/irrelevant context without abstaining)."

    return EvaluationRunRecord(
        system_name=system_name,
        test_case=test_case,
        answer=answer,
        iterations=iterations,
        snippets=result.snippets,
        is_correct=is_correct,
        is_grounded=is_grounded,
        is_unsupported=is_unsupported,
        is_abstained=is_abstained,
        matched_expected=matched_expected,
        latency_ms=round(latency_ms, 2),
        diagnostic=diagnostic,
    )


def compute_aggregate(
    system_name: str,
    records: Sequence[EvaluationRunRecord],
    recovery_gain: int | None = None,
) -> EvaluationAggregate:
    latencies = [r.latency_ms for r in records]
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    med_lat = float(statistics.median(latencies)) if latencies else 0.0

    return EvaluationAggregate(
        system_name=system_name,
        total_questions=len(records),
        correct_count=sum(1 for r in records if r.is_correct),
        grounded_count=sum(1 for r in records if r.is_grounded),
        unsupported_count=sum(1 for r in records if r.is_unsupported),
        abstained_count=sum(1 for r in records if r.is_abstained),
        total_iterations=sum(r.iterations for r in records),
        mean_latency_ms=round(mean_lat, 2),
        median_latency_ms=round(med_lat, 2),
        recovery_gain=recovery_gain,
    )


def compute_recovery_gain(
    b1_records: Sequence[EvaluationRunRecord],
    e1_records: Sequence[EvaluationRunRecord],
) -> dict[str, Any]:
    """Calculate the dataset-level Recovery Gain metric:

    Recovery Gain = (Newly Correct in E1) - (Newly Unsupported in E1)
    where:
    - Newly Correct: instances where E1 answered correctly while B1 failed or abstained.
    - Newly Unsupported: instances where E1 introduced an unsupported answer that B1 avoided.
    """
    newly_correct = 0
    newly_unsupported = 0
    details: list[str] = []

    for b1_rec, e1_rec in zip(b1_records, e1_records):
        tc_id = b1_rec.test_case.id
        if e1_rec.is_correct and not b1_rec.is_correct:
            newly_correct += 1
            details.append(f"[{tc_id}] Recovered correctly: E1 answered correctly while B1 failed/abstained.")
        if e1_rec.is_unsupported and not b1_rec.is_unsupported:
            newly_unsupported += 1
            details.append(f"[{tc_id}] Regression: E1 introduced unsupported answer where B1 avoided it.")

    gain = newly_correct - newly_unsupported
    return {
        "newly_correct": newly_correct,
        "newly_unsupported": newly_unsupported,
        "recovery_gain": gain,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Formatting & Export Helpers
# ---------------------------------------------------------------------------

def print_summary_table(aggregates: Sequence[EvaluationAggregate]) -> None:
    print("\n" + "=" * 105)
    print("EVALUATION SUMMARY TABLE (B0 Baseline vs B1 Sufficiency-Aware vs E1 EvidenceFirst)")
    print("=" * 105)
    header = (
        f"{'System':<20} | {'Questions':<9} | {'Correct':<10} | {'Grounded':<10} | "
        f"{'Unsupported':<11} | {'Abstained':<9} | {'Mean Lat':<9} | {'Avg Iter':<8} | {'Rec Gain':<8}"
    )
    print(header)
    print("-" * 105)

    for agg in aggregates:
        acc_str = f"{agg.correct_count} ({agg.accuracy * 100:.1f}%)"
        grd_str = f"{agg.grounded_count} ({agg.groundedness_rate * 100:.1f}%)"
        uns_str = f"{agg.unsupported_count} ({agg.unsupported_rate * 100:.1f}%)"
        abs_str = f"{agg.abstained_count} ({agg.abstention_rate * 100:.1f}%)"
        lat_str = f"{agg.mean_latency_ms:.1f} ms"
        rec_str = f"+{agg.recovery_gain}" if (agg.recovery_gain is not None and agg.recovery_gain > 0) else (str(agg.recovery_gain) if agg.recovery_gain is not None else "N/A")
        row = (
            f"{agg.system_name:<20} | {agg.total_questions:<9} | {acc_str:<10} | "
            f"{grd_str:<10} | {uns_str:<11} | {abs_str:<9} | {lat_str:<9} | {agg.average_iterations:<8.2f} | {rec_str:<8}"
        )
        print(row)
    print("=" * 105)


def print_per_question_results(
    records_by_case: Sequence[tuple[EvaluationRunRecord, EvaluationRunRecord, EvaluationRunRecord]],
) -> None:
    print("\n" + "=" * 95)
    print("PER-QUESTION DIAGNOSTIC RESULTS (B0 vs B1 vs E1)")
    print("=" * 95)

    for b0_rec, b1_rec, e1_rec in records_by_case:
        tc = b0_rec.test_case
        print(f"\n[{tc.id}] Question: {tc.question}")
        print(f"     Expected Label: {tc.expected_label.value.upper()}")
        print(f"     Description:    {tc.description}")
        print("     " + "-" * 90)

        for rec in (b0_rec, b1_rec, e1_rec):
            status_val = rec.answer.status.value if hasattr(rec.answer.status, "value") else str(rec.answer.status)
            score_str = f"{rec.answer.sufficiency_score:.2f}" if not rec.system_name.startswith("B0") else "N/A (no judge)"
            print(f"     [{rec.system_name}]")
            print(f"       Status:       {status_val}")
            print(f"       Iterations:   {rec.iterations}")
            print(f"       Latency:      {rec.latency_ms:.2f} ms")
            print(f"       Sufficiency:  {score_str}")
            print(f"       Answer:       \"{rec.answer.answer}\"")

            if rec.answer.citations:
                cites = [f"\"{c.claim}\" -> {', '.join(c.snippet_ids)}" for c in rec.answer.citations]
                print(f"       Citations:    {'; '.join(cites)}")
            else:
                print("       Citations:    None")

            if rec.answer.missing_facts:
                print(f"       Missing:      {', '.join(rec.answer.missing_facts)}")

            match_symbol = "✓ PASS" if rec.matched_expected else "✗ FAIL"
            print(f"       Evaluation:   {match_symbol} - {rec.diagnostic}")
        print("     " + "-" * 90)


def export_comparison_csvs(
    aggregates: Sequence[EvaluationAggregate],
    records_by_case: Sequence[tuple[EvaluationRunRecord, EvaluationRunRecord, EvaluationRunRecord]],
    output_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_comparison.csv"
    per_question_path = output_dir / "per_question_comparison.csv"

    # Export final comparison summary
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "system_name",
            "total_questions",
            "correct_count",
            "accuracy_pct",
            "grounded_count",
            "groundedness_rate_pct",
            "unsupported_count",
            "unsupported_rate_pct",
            "abstained_count",
            "abstention_rate_pct",
            "avg_iterations",
            "mean_latency_ms",
            "median_latency_ms",
            "recovery_gain",
        ])
        for agg in aggregates:
            rec_str = str(agg.recovery_gain) if agg.recovery_gain is not None else "N/A"
            writer.writerow([
                agg.system_name,
                agg.total_questions,
                agg.correct_count,
                round(agg.accuracy * 100, 2),
                agg.grounded_count,
                round(agg.groundedness_rate * 100, 2),
                agg.unsupported_count,
                round(agg.unsupported_rate * 100, 2),
                agg.abstained_count,
                round(agg.abstention_rate * 100, 2),
                round(agg.average_iterations, 2),
                agg.mean_latency_ms,
                agg.median_latency_ms,
                rec_str,
            ])

    # Export per-question trace
    with open(per_question_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "question",
            "expected_label",
            "b0_status",
            "b0_correct",
            "b0_unsupported",
            "b0_latency_ms",
            "b0_answer",
            "b1_status",
            "b1_correct",
            "b1_unsupported",
            "b1_latency_ms",
            "b1_answer",
            "e1_status",
            "e1_correct",
            "e1_unsupported",
            "e1_iterations",
            "e1_latency_ms",
            "e1_answer",
            "recovery_outcome",
        ])
        for b0, b1, e1 in records_by_case:
            tc = b0.test_case
            if e1.is_correct and not b1.is_correct:
                outcome = "recovered_correctly"
            elif e1.is_correct and b1.is_correct:
                outcome = "both_correct"
            elif e1.is_abstained and b1.is_abstained:
                outcome = "both_abstained_correctly"
            elif e1.is_unsupported:
                outcome = "regression_unsupported"
            else:
                outcome = "unrecovered_or_divergent"

            b0_status = b0.answer.status.value if hasattr(b0.answer.status, "value") else str(b0.answer.status)
            b1_status = b1.answer.status.value if hasattr(b1.answer.status, "value") else str(b1.answer.status)
            e1_status = e1.answer.status.value if hasattr(e1.answer.status, "value") else str(e1.answer.status)

            writer.writerow([
                tc.id,
                tc.question,
                tc.expected_label.value,
                b0_status,
                b0.is_correct,
                b0.is_unsupported,
                b0.latency_ms,
                b0.answer.answer,
                b1_status,
                b1.is_correct,
                b1.is_unsupported,
                b1.latency_ms,
                b1.answer.answer,
                e1_status,
                e1.is_correct,
                e1.is_unsupported,
                e1.iterations,
                e1.latency_ms,
                e1.answer.answer,
                outcome,
            ])

    return summary_path, per_question_path


# ---------------------------------------------------------------------------
# Main Evaluation Orchestration
# ---------------------------------------------------------------------------

def run_evaluation_suite(
    data_dir: Path = DATA_DIR,
    test_cases: Sequence[EvaluationTestCase] = BENCHMARK_TEST_CASES,
) -> tuple[EvaluationAggregate, EvaluationAggregate, EvaluationAggregate]:
    documents = load_documents(data_dir)
    corpora = tuple(
        Corpus(
            id=Path(doc.document_name).stem,
            description=f"{doc.document_name} {doc.text[:500]}",
        )
        for doc in documents
    )
    retriever = build_local_index(documents, chunk_size=800, overlap=120, per_query_limit=5)
    components = build_generic_components()

    # Systems initialization
    b0_system = B0ConventionalRAG(retriever)
    b1_system = B1SufficiencyAwareRAG(
        retriever=retriever,
        planner=components["planner"],
        rewriter=components["rewriter"],
        judge=components["judge"],
        drafter=components["drafter"],
        synthesizer=components["synthesizer"],
    )

    recovery_rewriter = ControlledRecoveryRewriter(components["rewriter"], default_mask_revenue=False)
    e1_orchestrator = AgenticRAGOrchestrator(
        planner=components["planner"],
        rewriter=recovery_rewriter,
        retriever=retriever,
        drafter=components["drafter"],
        judge=components["judge"],
        synthesizer=components["synthesizer"],
        config=OrchestratorConfig(max_iterations=2),
    )

    b0_records: list[EvaluationRunRecord] = []
    b1_records: list[EvaluationRunRecord] = []
    e1_records: list[EvaluationRunRecord] = []
    paired_records: list[tuple[EvaluationRunRecord, EvaluationRunRecord, EvaluationRunRecord]] = []

    for test_case in test_cases:
        # B0 run
        t0 = time.perf_counter()
        b0_res = b0_system.run(test_case.question, corpora, mask=test_case.initial_corpus_mask)
        b0_lat = (time.perf_counter() - t0) * 1000.0
        b0_rec = evaluate_run_record(test_case, b0_res, "B0 Baseline", latency_ms=b0_lat)
        b0_records.append(b0_rec)

        # B1 run
        t1 = time.perf_counter()
        b1_res = b1_system.run(test_case.question, corpora, mask=test_case.initial_corpus_mask)
        b1_lat = (time.perf_counter() - t1) * 1000.0
        b1_rec = evaluate_run_record(test_case, b1_res, "B1 Sufficiency-Aware", latency_ms=b1_lat)
        b1_records.append(b1_rec)

        # E1 run
        recovery_rewriter.set_mask(test_case.initial_corpus_mask)
        t2 = time.perf_counter()
        e1_res = e1_orchestrator.run(test_case.question, corpora)
        e1_lat = (time.perf_counter() - t2) * 1000.0
        e1_rec = evaluate_run_record(test_case, e1_res, "E1 EvidenceFirst", latency_ms=e1_lat)
        e1_records.append(e1_rec)

        paired_records.append((b0_rec, b1_rec, e1_rec))

    # Calculate metrics
    rec_gain_info = compute_recovery_gain(b1_records, e1_records)
    b0_aggregate = compute_aggregate("B0 Baseline", b0_records, recovery_gain=None)
    b1_aggregate = compute_aggregate("B1 Sufficiency-Aware", b1_records, recovery_gain=None)
    e1_aggregate = compute_aggregate("E1 EvidenceFirst", e1_records, recovery_gain=rec_gain_info["recovery_gain"])

    # Output reports and CSVs
    print_summary_table((b0_aggregate, b1_aggregate, e1_aggregate))
    print_per_question_results(paired_records)

    print("\nRECOVERY GAIN METRIC ANALYSIS (E1 vs B1)")
    print("-" * 60)
    print(f"Newly Correct Answers (recovered by E1):  {rec_gain_info['newly_correct']}")
    print(f"Newly Unsupported Answers (regressions):  {rec_gain_info['newly_unsupported']}")
    print(f"Dataset-Level Net Recovery Gain:          +{rec_gain_info['recovery_gain']}")
    for det in rec_gain_info["details"]:
        print(f"  * {det}")

    summary_csv, per_q_csv = export_comparison_csvs((b0_aggregate, b1_aggregate, e1_aggregate), paired_records)
    print(f"\n✓ Exported final comparison table to: {summary_csv}")
    print(f"✓ Exported per-question comparison to: {per_q_csv}")

    return b0_aggregate, b1_aggregate, e1_aggregate


def main() -> None:
    run_evaluation_suite()


if __name__ == "__main__":
    main()
