"""Adaptive Recovery Controller for EvidenceFirst RAG (Person 2 — Darshit).

Implements single-pass targeted query reformulation and recovery retrieval:
- Receives question, evidence chunks, and missing_information from sufficiency judge
- Generates ONE focused search query
- Calls retrieval exactly ONE additional time (max_attempts = 1)
- Merges/deduplicates evidence
- Re-runs sufficiency evaluation
- If still insufficient, returns transparent abstention
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from evidencefirst_rag.chunking import Chunk


@dataclass
class RecoveryTrace:
    """Trace details for the adaptive recovery attempt."""

    triggered: bool = False
    original_missing_facts: list[str] = field(default_factory=list)
    reformulated_query: str = ""
    recovered_chunks: list[Chunk] = field(default_factory=list)
    post_recovery_sufficiency: bool = False
    post_recovery_probability: float = 0.0
    attempts: int = 0


class AdaptiveRecoveryController:
    """Controls bounded recovery retrieval when initial context is insufficient."""

    def __init__(self, max_recovery_attempts: int = 1) -> None:
        if max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        self.max_recovery_attempts = max_recovery_attempts

    def reformulate_query(self, question: str, missing_information: Sequence[str]) -> str:
        """Formulate a single focused search query targeting the missing facts."""
        if not missing_information:
            return question.strip()

        # Combine question keywords with missing information focus
        stopwords = {
            "what", "who", "when", "where", "why", "how",
            "is", "are", "was", "were", "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "by", "with",
            "missing", "context", "fact", "required", "evidence",
        }

        # Extract salient terms from missing information
        missing_terms: list[str] = []
        for fact in missing_information:
            tokens = re.findall(r"[a-z0-9]+", str(fact).lower())
            for t in tokens:
                if t not in stopwords and t not in missing_terms:
                    missing_terms.append(t)

        # Extract entity terms from original question
        q_tokens = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in stopwords]

        # Interleave question entity anchor with missing attributes
        combined = list(dict.fromkeys(q_tokens[:3] + missing_terms[:4]))
        reformulated = " ".join(combined).strip()

        return reformulated or question.strip()

    def execute_recovery(
        self,
        question: str,
        initial_chunks: Sequence[Chunk],
        missing_information: Sequence[str],
        retriever_func: Any,
        judge_func: Any,
    ) -> tuple[list[Chunk], Any, RecoveryTrace]:
        """Execute exactly one recovery retrieval pass and re-evaluate sufficiency."""
        if self.max_recovery_attempts == 0:
            trace = RecoveryTrace(
                triggered=False,
                original_missing_facts=list(missing_information),
                attempts=0,
            )
            return list(initial_chunks), judge_func(question, list(initial_chunks)), trace

        trace = RecoveryTrace(
            triggered=True,
            original_missing_facts=list(missing_information),
            attempts=1,
        )

        reformulated_query = self.reformulate_query(question, missing_information)
        trace.reformulated_query = reformulated_query

        # Retrieve one additional time
        new_chunks: list[Chunk] = retriever_func(reformulated_query, top_k=6)
        trace.recovered_chunks = list(new_chunks)

        # Merge and deduplicate evidence chunks
        seen_ids = set()
        merged_chunks: list[Chunk] = []
        for c in list(initial_chunks) + new_chunks:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                merged_chunks.append(c)

        # Re-run sufficiency evaluation
        re_assessment = judge_func(question, merged_chunks)
        trace.post_recovery_sufficiency = bool(getattr(re_assessment, "sufficient", False))
        trace.post_recovery_probability = float(getattr(re_assessment, "probability", 0.0))

        return merged_chunks, re_assessment, trace
