"""End-to-End EvidenceFirst RAG Pipeline (Master Plan Target Architecture).

Integrates all 3 roles:
- Person 1: Ingestion, Chunking (~550t/~80o), Dense + BM25 -> RRF -> Cross-Encoder -> Top 6
- Person 3: Context Sufficiency Evaluator (Diligent Reader, probability, missing facts)
- Person 2: Adaptive Recovery Controller (max 1 pass) -> Citation Verifier -> Grounded Answer / Abstention
- UI Trace & Experiment Logging
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.generation import GroundedAnswerRecord, GroundedGenerator
from evidencefirst_rag.recovery import AdaptiveRecoveryController, RecoveryTrace
from evidencefirst_rag.retrieval import HybridRetriever
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator, StructuredSufficiencyResult
from evidencefirst_rag.verifier import CitationVerifier, VerificationReport


@dataclass
class PipelineTrace:
    """Complete auditable execution trace of a question through the pipeline."""

    question: str
    initial_query: str
    retrieved_chunks: list[Chunk] = field(default_factory=list)
    sufficiency_result: StructuredSufficiencyResult | None = None
    recovery_trace: RecoveryTrace | None = None
    final_evidence_chunks: list[Chunk] = field(default_factory=list)
    answer_record: GroundedAnswerRecord | None = None
    verification_report: VerificationReport | None = None
    final_decision: str = ""  # 'ANSWERED' or 'ABSTAINED'
    final_answer: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "initial_query": self.initial_query,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "sufficiency_result": self.sufficiency_result.to_dict() if self.sufficiency_result else None,
            "recovery_trace": {
                "triggered": self.recovery_trace.triggered,
                "reformulated_query": self.recovery_trace.reformulated_query,
                "original_missing_facts": self.recovery_trace.original_missing_facts,
                "recovered_chunks_count": len(self.recovery_trace.recovered_chunks),
                "post_recovery_sufficiency": self.recovery_trace.post_recovery_sufficiency,
            } if self.recovery_trace else None,
            "final_evidence_count": len(self.final_evidence_chunks),
            "answer_record": self.answer_record.to_dict() if self.answer_record else None,
            "verification_report": self.verification_report.to_dict() if self.verification_report else None,
            "final_decision": self.final_decision,
            "final_answer": self.final_answer,
            "latency_ms": round(self.latency_ms, 2),
        }


class EvidenceFirstPipeline:
    """Master EvidenceFirst RAG Orchestrator."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        retriever: HybridRetriever | None = None,
        evaluator: ContextSufficiencyEvaluator | None = None,
        recovery_controller: AdaptiveRecoveryController | None = None,
        generator: GroundedGenerator | None = None,
        verifier: CitationVerifier | None = None,
        autorater: Any | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.retriever = retriever or HybridRetriever(self.chunks)
        if evaluator is not None:
            self.evaluator = evaluator
        elif autorater is not None:
            self.evaluator = ContextSufficiencyEvaluator(autorater=autorater)
        else:
            self.evaluator = ContextSufficiencyEvaluator()
        self.recovery_controller = recovery_controller or AdaptiveRecoveryController(max_recovery_attempts=1)
        self.generator = generator or GroundedGenerator()
        self.verifier = verifier or CitationVerifier()

    def run(self, question: str) -> PipelineTrace:
        """Execute the full end-to-end EvidenceFirst workflow on a user question."""
        start_time = time.perf_counter()

        trace = PipelineTrace(
            question=question,
            initial_query=question,
        )

        # Step 1: Hybrid Retrieval -> RRF -> Cross-Encoder -> Top 6 Evidence Chunks
        top_chunks = self.retriever.retrieve(question, top_k=6, mode="hybrid_rerank")
        trace.retrieved_chunks = list(top_chunks)
        trace.final_evidence_chunks = list(top_chunks)

        # Step 2: Context Sufficiency Evaluation
        sufficiency_res = self.evaluator.evaluate(question, top_chunks)
        trace.sufficiency_result = sufficiency_res

        current_chunks = list(top_chunks)
        is_sufficient = sufficiency_res.sufficient

        # Step 3: Adaptive Recovery Branch (if insufficient)
        if not is_sufficient:
            # Execute exactly ONE recovery pass
            recovered_chunks, post_eval, recovery_trace = self.recovery_controller.execute_recovery(
                question=question,
                initial_chunks=top_chunks,
                missing_information=sufficiency_res.missing_information,
                retriever_func=lambda q, top_k: self.retriever.retrieve(q, top_k=top_k, mode="hybrid_rerank"),
                judge_func=lambda q, ch: self.evaluator.evaluate(q, ch),
            )
            trace.recovery_trace = recovery_trace
            trace.final_evidence_chunks = recovered_chunks
            current_chunks = recovered_chunks
            is_sufficient = recovery_trace.post_recovery_sufficiency

        # Step 4: Grounded Answer Drafting
        answer_record = self.generator.generate(
            question=question,
            evidence_chunks=current_chunks,
            is_sufficient=is_sufficient,
        )
        trace.answer_record = answer_record

        # Step 5: Claim Support Verification
        verif_report = self.verifier.verify_claims(answer_record, current_chunks)
        trace.verification_report = verif_report

        # Step 6: Final Decision (Answer or Abstain)
        if is_sufficient and verif_report.all_supported and not answer_record.is_abstained:
            trace.final_decision = "ANSWERED"
            trace.final_answer = answer_record.answer
        else:
            trace.final_decision = "ABSTAINED"
            trace.final_answer = "The answer cannot be established from the available corpus."

        trace.latency_ms = (time.perf_counter() - start_time) * 1000.0
        return trace
