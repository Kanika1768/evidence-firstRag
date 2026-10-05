"""Evaluation and Selective Answering Module for EvidenceFirst RAG (Person 3 — Kanika).

Implements:
1. Selective Answering Policy:
   - Fits a calibrated logistic model on the development split:
     P(correct | sufficient, confidence) = sigma(w1 * p_suff + w2 * conf + b)
   - Evaluates across thresholds tau in [0.0, 1.0] to compute:
     - Coverage at >= 90% accuracy
     - Coverage at >= 95% accuracy
     - Area Under the Accuracy-Coverage Curve (AUACC)
     - Rejection Quality
2. Baseline Comparison Harness (B0 vs B1 vs E1):
   - B0 Baseline (Conventional 1-shot RAG, no abstention)
   - B1 Sufficiency-Aware RAG (1-shot with sufficiency judgment + selective abstention)
   - E1 EvidenceFirst RAG (Hybrid retrieval + sufficiency judge + adaptive recovery + verifier)
   - Recovery Gain calculation: (Newly Correct in E1) - (Newly Unsupported in E1)
"""

from __future__ import annotations

import csv
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.generation import GroundedAnswerRecord
from evidencefirst_rag.pipeline import EvidenceFirstPipeline, PipelineTrace
from evidencefirst_rag.retrieval import HybridRetriever
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator, StructuredSufficiencyResult
from evidencefirst_rag.verifier import CitationVerifier


@dataclass
class SelectiveAnsweringModel:
    """Logistic regression model for selective answering: P(correct | p_suff, conf)."""

    w1: float = 2.5
    w2: float = 1.5
    b: float = -2.0

    def predict_proba(self, p_suff: float, conf: float) -> float:
        """Compute calibrated probability of correctness."""
        z = self.w1 * float(p_suff) + self.w2 * float(conf) + self.b
        # Clamp to avoid numerical overflow
        z = max(-25.0, min(25.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(
        self,
        samples: Sequence[tuple[float, float]],
        labels: Sequence[int],
        epochs: int = 250,
        lr: float = 0.1,
    ) -> None:
        """Fit weights using binary cross-entropy gradient descent."""
        if not samples or len(samples) != len(labels):
            return

        n = len(samples)
        for _ in range(epochs):
            gw1, gw2, gb = 0.0, 0.0, 0.0
            for (p_s, c), y in zip(samples, labels):
                p_hat = self.predict_proba(p_s, c)
                err = p_hat - y
                gw1 += err * p_s
                gw2 += err * c
                gb += err
            self.w1 -= lr * (gw1 / n)
            self.w2 -= lr * (gw2 / n)
            self.b -= lr * (gb / n)


@dataclass
class AccuracyCoveragePoint:
    threshold: float
    coverage: float
    accuracy: float
    attempted_count: int
    correct_count: int
    total_count: int


@dataclass
class AccuracyCoverageReport:
    points: list[AccuracyCoveragePoint]
    coverage_at_90_acc: float
    coverage_at_95_acc: float
    auacc: float
    rejection_quality: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_at_90_pct_accuracy": round(self.coverage_at_90_acc, 4),
            "coverage_at_95_pct_accuracy": round(self.coverage_at_95_acc, 4),
            "auacc": round(self.auacc, 4),
            "rejection_quality": round(self.rejection_quality, 4),
            "curve_points": [
                {
                    "threshold": round(p.threshold, 3),
                    "coverage": round(p.coverage, 4),
                    "accuracy": round(p.accuracy, 4),
                    "attempted": p.attempted_count,
                    "correct": p.correct_count,
                }
                for p in self.points
            ],
        }


class SelectiveAnsweringPolicy:
    """Manages thresholding and selective answering evaluation."""

    def __init__(self, model: SelectiveAnsweringModel | None = None) -> None:
        self.model = model or SelectiveAnsweringModel()

    def decide_to_answer(self, p_suff: float, conf: float, tau: float = 0.5) -> bool:
        """Decide whether the model should attempt an answer at threshold tau."""
        prob = self.model.predict_proba(p_suff, conf)
        return prob >= tau

    def evaluate_curve(
        self,
        records: Sequence[tuple[float, float, bool]],  # (p_suff, conf, is_correct)
        num_thresholds: int = 50,
    ) -> AccuracyCoverageReport:
        """Sweep threshold tau in [0.0, 1.0] and compute the accuracy-coverage curve."""
        if not records:
            return AccuracyCoverageReport([], 0.0, 0.0, 0.0, 0.0)

        total_n = len(records)
        overall_accuracy = sum(1 for _, _, corr in records if corr) / total_n

        points: list[AccuracyCoveragePoint] = []
        for i in range(num_thresholds + 1):
            tau = i / float(num_thresholds)
            attempted = [r for r in records if self.model.predict_proba(r[0], r[1]) >= tau]
            att_n = len(attempted)
            coverage = att_n / total_n if total_n else 0.0
            corr_n = sum(1 for _, _, corr in attempted if corr)
            acc = corr_n / att_n if att_n else 1.0
            points.append(
                AccuracyCoveragePoint(
                    threshold=tau,
                    coverage=coverage,
                    accuracy=acc,
                    attempted_count=att_n,
                    correct_count=corr_n,
                    total_count=total_n,
                )
            )

        # Sort points by increasing coverage
        points.sort(key=lambda p: p.coverage)

        # Coverage at 90% and 95% accuracy
        cov_90 = 0.0
        cov_95 = 0.0
        for p in points:
            if p.accuracy >= 0.90 and p.coverage > cov_90:
                cov_90 = p.coverage
            if p.accuracy >= 0.95 and p.coverage > cov_95:
                cov_95 = p.coverage

        # Compute AUACC via trapezoidal integration over coverage [0.0, 1.0]
        auacc = 0.0
        for idx in range(len(points) - 1):
            p1 = points[idx]
            p2 = points[idx + 1]
            dcov = p2.coverage - p1.coverage
            if dcov > 0:
                avg_acc = (p1.accuracy + p2.accuracy) / 2.0
                auacc += avg_acc * dcov

        # Rejection quality: accuracy gain on answered questions at tau=0.5
        tau_05_attempted = [r for r in records if self.model.predict_proba(r[0], r[1]) >= 0.5]
        tau_05_acc = (
            sum(1 for _, _, corr in tau_05_attempted if corr) / len(tau_05_attempted)
            if tau_05_attempted
            else 1.0
        )
        rejection_quality = tau_05_acc - overall_accuracy

        return AccuracyCoverageReport(
            points=points,
            coverage_at_90_acc=cov_90,
            coverage_at_95_acc=cov_95,
            auacc=auacc,
            rejection_quality=rejection_quality,
        )


@dataclass
class BaselineQuestion:
    """Evaluation question definition."""

    id: str
    question: str
    is_answerable: bool
    expected_terms: tuple[str, ...] = ()
    description: str = ""
    mask_on_pass_1: tuple[str, ...] = ()


@dataclass
class BaselineRunOutput:
    system_name: str
    question_id: str
    question: str
    is_answerable: bool
    status: str
    answer: str
    is_correct: bool
    is_grounded: bool
    is_unsupported: bool
    is_abstained: bool
    iterations: int
    latency_ms: float


@dataclass
class BaselineSystemSummary:
    system_name: str
    total_questions: int
    correct_count: int
    grounded_count: int
    unsupported_count: int
    abstained_count: int
    total_iterations: int
    mean_latency_ms: float
    median_latency_ms: float
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
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total_questions if self.total_questions else 0.0


# Default NovaTech benchmark evaluation suite
EVALUATION_QUESTIONS: tuple[BaselineQuestion, ...] = (
    BaselineQuestion(
        id="Q1",
        question="Who founded NovaTech and when was it founded?",
        is_answerable=True,
        expected_terms=("priya mehta", "2019"),
        description="Multi-corpus fact requirement: founder and year.",
    ),
    BaselineQuestion(
        id="Q2",
        question="What does NovaTech develop?",
        is_answerable=True,
        expected_terms=("analytics software",),
        description="Single-corpus product factual question.",
    ),
    BaselineQuestion(
        id="Q3",
        question="What is NovaTech's annual revenue?",
        is_answerable=True,
        expected_terms=("42 million",),
        description="Single-corpus financial question.",
    ),
    BaselineQuestion(
        id="Q4",
        question="Who is the CEO of NovaTech?",
        is_answerable=False,
        expected_terms=(),
        description="Unanswerable question: CEO info is completely absent.",
    ),
    BaselineQuestion(
        id="Q5",
        question="What is NovaTech's stock price?",
        is_answerable=False,
        expected_terms=(),
        description="Unanswerable question: stock price is completely absent.",
    ),
    BaselineQuestion(
        id="Q6",
        question="How many employees does NovaTech have?",
        is_answerable=False,
        expected_terms=(),
        description="Unanswerable question: employee count is completely absent.",
    ),
    BaselineQuestion(
        id="Q7",
        question="What is NovaTech's annual revenue?",
        is_answerable=True,
        expected_terms=("42 million",),
        mask_on_pass_1=("finance",),
        description="Controlled recovery: finance masked on pass 1, recovered on pass 2.",
    ),
)


class BaselineEvaluator:
    """Orchestrates comparative evaluation across B0, B1, and E1."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.retriever = HybridRetriever(self.chunks)
        self.evaluator = ContextSufficiencyEvaluator()
        self.pipeline = EvidenceFirstPipeline(self.chunks)

    def run_b0(self, q: BaselineQuestion) -> BaselineRunOutput:
        """Run B0 Conventional One-Shot RAG (no judge, always answers)."""
        t0 = time.perf_counter()
        # Search chunks excluding masked docs if requested
        search_chunks = (
            [c for c in self.chunks if c.document_id not in q.mask_on_pass_1]
            if q.mask_on_pass_1
            else self.chunks
        )
        r = HybridRetriever(search_chunks)
        top = r.retrieve(q.question, top_k=6, mode="hybrid_rerank")
        lat = (time.perf_counter() - t0) * 1000.0

        if top:
            ans_text = top[0].text
        else:
            ans_text = "No documents found."

        # B0 always attempts to answer
        ans_lower = ans_text.lower()
        if q.is_answerable:
            is_corr = all(term.lower() in ans_lower for term in q.expected_terms)
            is_unsup = False
        else:
            # Unanswerable question answered by B0 -> unsupported hallucination!
            is_corr = False
            is_unsup = True

        return BaselineRunOutput(
            system_name="B0 Baseline",
            question_id=q.id,
            question=q.question,
            is_answerable=q.is_answerable,
            status="ANSWERED",
            answer=ans_text,
            is_correct=is_corr,
            is_grounded=is_corr,
            is_unsupported=is_unsup,
            is_abstained=False,
            iterations=1,
            latency_ms=lat,
        )

    def run_b1(self, q: BaselineQuestion) -> BaselineRunOutput:
        """Run B1 Sufficiency-Aware One-Shot RAG (abstains if insufficient, 0 recovery)."""
        t0 = time.perf_counter()
        search_chunks = (
            [c for c in self.chunks if c.document_id not in q.mask_on_pass_1]
            if q.mask_on_pass_1
            else self.chunks
        )
        r = HybridRetriever(search_chunks)
        top = r.retrieve(q.question, top_k=6, mode="hybrid_rerank")
        suff = self.evaluator.evaluate(q.question, top)

        if suff.sufficient:
            # Generate grounded answer
            gen_rec = self.pipeline.generator.generate(q.question, top, is_sufficient=True)
            status = "ANSWERED"
            ans_text = gen_rec.answer
            is_abstained = False
        else:
            status = "ABSTAINED"
            ans_text = "The answer cannot be established from the available corpus."
            is_abstained = True

        lat = (time.perf_counter() - t0) * 1000.0
        ans_lower = ans_text.lower()

        if q.is_answerable:
            is_corr = not is_abstained and all(term.lower() in ans_lower for term in q.expected_terms)
            is_unsup = False
        else:
            is_corr = is_abstained
            is_unsup = not is_abstained

        return BaselineRunOutput(
            system_name="B1 Sufficiency-Aware",
            question_id=q.id,
            question=q.question,
            is_answerable=q.is_answerable,
            status=status,
            answer=ans_text,
            is_correct=is_corr,
            is_grounded=is_corr,
            is_unsupported=is_unsup,
            is_abstained=is_abstained,
            iterations=1,
            latency_ms=lat,
        )

    def run_e1(self, q: BaselineQuestion) -> BaselineRunOutput:
        """Run E1 EvidenceFirst Adaptive RAG (hybrid + recovery + verifier)."""
        t0 = time.perf_counter()
        if q.mask_on_pass_1:
            # Custom pass 1 with mask, then pass 2 with recovery across full corpus
            search_chunks = [c for c in self.chunks if c.document_id not in q.mask_on_pass_1]
            r1 = HybridRetriever(search_chunks)
            top1 = r1.retrieve(q.question, top_k=6, mode="hybrid_rerank")
            suff1 = self.evaluator.evaluate(q.question, top1)

            # Recovery pass against full corpus
            recovered_chunks, post_eval, recovery_trace = self.pipeline.recovery_controller.execute_recovery(
                question=q.question,
                initial_chunks=top1,
                missing_information=suff1.missing_information,
                retriever_func=lambda query, top_k: self.retriever.retrieve(query, top_k=top_k, mode="hybrid_rerank"),
                judge_func=lambda query, ch: self.evaluator.evaluate(query, ch),
            )
            answer_record = self.pipeline.generator.generate(
                question=q.question,
                evidence_chunks=recovered_chunks,
                is_sufficient=recovery_trace.post_recovery_sufficiency,
            )
            verif = self.pipeline.verifier.verify_claims(answer_record, recovered_chunks)
            if recovery_trace.post_recovery_sufficiency and verif.all_supported:
                status = "ANSWERED"
                ans_text = answer_record.answer
                is_abstained = False
            else:
                status = "ABSTAINED"
                ans_text = "The answer cannot be established from the available corpus."
                is_abstained = True
            iterations = 2
        else:
            trace = self.pipeline.run(q.question)
            status = trace.final_decision
            ans_text = trace.final_answer
            is_abstained = trace.final_decision == "ABSTAINED"
            iterations = 2 if trace.recovery_trace and trace.recovery_trace.triggered else 1

        lat = (time.perf_counter() - t0) * 1000.0
        ans_lower = ans_text.lower()

        if q.is_answerable:
            is_corr = not is_abstained and all(term.lower() in ans_lower for term in q.expected_terms)
            is_unsup = False
        else:
            is_corr = is_abstained
            is_unsup = not is_abstained

        return BaselineRunOutput(
            system_name="E1 EvidenceFirst",
            question_id=q.id,
            question=q.question,
            is_answerable=q.is_answerable,
            status=status,
            answer=ans_text,
            is_correct=is_corr,
            is_grounded=is_corr,
            is_unsupported=is_unsup,
            is_abstained=is_abstained,
            iterations=iterations,
            latency_ms=lat,
        )

    def run_suite(
        self,
        questions: Sequence[BaselineQuestion] = EVALUATION_QUESTIONS,
    ) -> tuple[BaselineSystemSummary, BaselineSystemSummary, BaselineSystemSummary]:
        """Execute full evaluation comparing B0, B1, and E1."""
        b0_records = [self.run_b0(q) for q in questions]
        b1_records = [self.run_b1(q) for q in questions]
        e1_records = [self.run_e1(q) for q in questions]

        # Calculate Recovery Gain: (Newly Correct in E1) - (Newly Unsupported in E1)
        newly_correct = sum(
            1 for b1_r, e1_r in zip(b1_records, e1_records) if e1_r.is_correct and not b1_r.is_correct
        )
        newly_unsupported = sum(
            1 for b1_r, e1_r in zip(b1_records, e1_records) if e1_r.is_unsupported and not b1_r.is_unsupported
        )
        rec_gain = newly_correct - newly_unsupported

        def _summarize(name: str, records: list[BaselineRunOutput], rg: int | None = None) -> BaselineSystemSummary:
            lats = [r.latency_ms for r in records]
            return BaselineSystemSummary(
                system_name=name,
                total_questions=len(records),
                correct_count=sum(1 for r in records if r.is_correct),
                grounded_count=sum(1 for r in records if r.is_grounded),
                unsupported_count=sum(1 for r in records if r.is_unsupported),
                abstained_count=sum(1 for r in records if r.is_abstained),
                total_iterations=sum(r.iterations for r in records),
                mean_latency_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
                median_latency_ms=round(statistics.median(lats), 2) if lats else 0.0,
                recovery_gain=rg,
            )

        b0_sum = _summarize("B0 Baseline", b0_records, None)
        b1_sum = _summarize("B1 Sufficiency-Aware", b1_records, None)
        e1_sum = _summarize("E1 EvidenceFirst", e1_records, rec_gain)

        return b0_sum, b1_sum, e1_sum
