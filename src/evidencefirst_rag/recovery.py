"""Adaptive Recovery Controller for EvidenceFirst RAG (Person 2 — Darshit & Extensions).

Implements bounded multi-step cross-document evidence recovery:
- Receives question, evidence chunks, and missing_information from sufficiency judge
- Dynamically reformulates queries using intermediate bridge entities across documents
- Executes configurable multi-step recovery hops (K=1, K=2, K=3)
- Features cycle detection (preventing repetitive queries)
- Deduplicates chunks and tracks evidence provenance
- Terminates early upon sufficient convergence, cycle detection, or evidence stagnation
- Transparently reports multi-hop trace details
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from evidencefirst_rag.chunking import Chunk


@dataclass
class HopTrace:
    """Trace details for an individual recovery hop."""

    hop_number: int
    target_missing_facts: list[str]
    formulated_query: str
    retrieved_chunk_ids: list[str]
    new_unique_chunk_ids: list[str]
    sufficiency_after_hop: bool
    probability_after_hop: float
    source_documents: list[str]


@dataclass
class RecoveryTrace:
    """Comprehensive trace details for adaptive recovery execution."""

    triggered: bool = False
    original_missing_facts: list[str] = field(default_factory=list)
    reformulated_query: str = ""
    recovered_chunks: list[Chunk] = field(default_factory=list)
    post_recovery_sufficiency: bool = False
    post_recovery_probability: float = 0.0
    attempts: int = 0
    # Multi-step extensions
    max_depth: int = 1
    total_hops_executed: int = 0
    hop_traces: list[HopTrace] = field(default_factory=list)
    cycle_detected: bool = False
    termination_reason: str = "COMPLETED"
    provenance_chain: list[str] = field(default_factory=list)


class AdaptiveRecoveryController:
    """Controls bounded multi-step recovery retrieval when initial context is insufficient."""

    def __init__(self, max_recovery_attempts: int = 1, max_depth: int | None = None) -> None:
        depth = max_depth if max_depth is not None else max_recovery_attempts
        if depth < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        self.max_recovery_attempts = depth
        self.max_depth = depth

    @staticmethod
    def extract_bridge_entities(chunks: Sequence[Chunk], missing_information: Sequence[str]) -> list[str]:
        """Extract candidate bridge entities from intermediate evidence chunks using heuristic regex rules.

        Note: This is an explicit heuristic pattern-matching extraction targeting capitalized proper nouns
        and standard alphanumeric identifiers, rather than an exhaustive or semantic NER extractor.
        """
        bridge_entities: list[str] = []
        # Identify capitalized entities, quoted terms, and model numbers from recent chunks
        for c in chunks:
            # Proper nouns / capitalized sequences (e.g. "Global AI Institute", "Turing Medallion")
            entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", c.text)
            for ent in entities:
                ent_clean = ent.strip()
                if len(ent_clean) > 3 and ent_clean not in bridge_entities:
                    bridge_entities.append(ent_clean)

            # Technical codes / standard numbers (e.g. "SP 800-61r3", "AI-100-1")
            codes = re.findall(r"\b(?:[A-Z]{2,}[-\s]?[0-9]+[a-zA-Z0-9\-]*)\b", c.text)
            for code in codes:
                code_clean = code.strip()
                if code_clean not in bridge_entities:
                    bridge_entities.append(code_clean)

        return bridge_entities[:5]

    def reformulate_query(
        self,
        question: str,
        missing_information: Sequence[str],
        intermediate_chunks: Sequence[Chunk] = (),
    ) -> str:
        """Formulate a targeted search query combining question anchors, missing facts, and bridge entities."""
        if not missing_information:
            return question.strip()

        stopwords = {
            "what", "who", "when", "where", "why", "how",
            "is", "are", "was", "were", "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "by", "with",
            "missing", "context", "fact", "required", "evidence", "information",
        }

        # 1. Extract salient terms from missing information
        missing_terms: list[str] = []
        for fact in missing_information:
            tokens = re.findall(r"[a-z0-9\-]+", str(fact).lower())
            for t in tokens:
                if t not in stopwords and t not in missing_terms and len(t) > 1:
                    missing_terms.append(t)

        # 2. Extract entity terms from original question
        q_tokens = [t for t in re.findall(r"[a-z0-9\-]+", question.lower()) if t not in stopwords and len(t) > 1]

        # 3. Extract bridge entities from intermediate chunks
        bridge_terms: list[str] = []
        if intermediate_chunks:
            bridges = self.extract_bridge_entities(intermediate_chunks, missing_information)
            for b in bridges:
                b_toks = [t for t in re.findall(r"[a-z0-9\-]+", b.lower()) if t not in stopwords and len(t) > 1]
                for bt in b_toks:
                    if bt not in bridge_terms and bt not in q_tokens:
                        bridge_terms.append(bt)

        # Prioritize bridge entities + missing terms + core question anchor
        combined = list(dict.fromkeys(bridge_terms[:3] + missing_terms[:4] + q_tokens[:2]))
        reformulated = " ".join(combined).strip()

        return reformulated or question.strip()

    def execute_recovery(
        self,
        question: str,
        initial_chunks: Sequence[Chunk],
        missing_information: Sequence[str],
        retriever_func: Any,
        judge_func: Any,
        max_depth: int | None = None,
    ) -> tuple[list[Chunk], Any, RecoveryTrace]:
        """Execute bounded multi-step recovery retrieval up to configured depth."""
        effective_depth = max_depth if max_depth is not None else self.max_recovery_attempts
        if effective_depth < 0:
            raise ValueError("max_depth must be non-negative")

        if effective_depth == 0:
            trace = RecoveryTrace(
                triggered=False,
                original_missing_facts=list(missing_information),
                attempts=0,
                max_depth=0,
                total_hops_executed=0,
                termination_reason="ZERO_ATTEMPTS",
            )
            return list(initial_chunks), judge_func(question, list(initial_chunks)), trace

        trace = RecoveryTrace(
            triggered=True,
            original_missing_facts=list(missing_information),
            max_depth=effective_depth,
        )

        seen_chunk_ids: set[str] = {c.chunk_id for c in initial_chunks}
        seen_queries: set[str] = set()
        current_chunks: list[Chunk] = list(initial_chunks)
        current_missing: list[str] = list(missing_information)
        last_assessment: Any = None

        for hop in range(1, effective_depth + 1):
            # Formulate next-hop query targeting remaining gaps
            reformulated_query = self.reformulate_query(
                question=question,
                missing_information=current_missing,
                intermediate_chunks=current_chunks,
            )

            # Cycle detection: check if this reformulated query repeats an earlier hop
            query_key = " ".join(sorted(set(re.findall(r"\w+", reformulated_query.lower()))))
            if query_key in seen_queries:
                trace.cycle_detected = True
                trace.termination_reason = "CYCLE_DETECTED"
                break
            seen_queries.add(query_key)

            if hop == 1:
                trace.reformulated_query = reformulated_query

            # Count hop only when retrieval call is actually executed
            trace.total_hops_executed = hop
            trace.attempts = hop

            # Execute recovery retrieval pass
            retrieved_chunks: list[Chunk] = retriever_func(reformulated_query, top_k=6)
            new_chunks = [c for c in retrieved_chunks if c.chunk_id not in seen_chunk_ids]

            # Provenance tracking
            hop_docs = sorted({c.document_name for c in retrieved_chunks if getattr(c, "document_name", None)})

            # Evidence stagnation check
            if not new_chunks:
                trace.termination_reason = "NO_NEW_CHUNKS"
                hop_trace = HopTrace(
                    hop_number=hop,
                    target_missing_facts=list(current_missing),
                    formulated_query=reformulated_query,
                    retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
                    new_unique_chunk_ids=[],
                    sufficiency_after_hop=False,
                    probability_after_hop=0.0,
                    source_documents=hop_docs,
                )
                trace.hop_traces.append(hop_trace)
                break

            # Integrate new evidence
            for c in new_chunks:
                seen_chunk_ids.add(c.chunk_id)
                current_chunks.append(c)
                trace.recovered_chunks.append(c)

            # Re-run sufficiency evaluation
            re_assessment = judge_func(question, current_chunks)
            last_assessment = re_assessment

            is_suff = bool(getattr(re_assessment, "sufficient", False))
            prob = float(getattr(re_assessment, "probability", 0.0))
            new_missing = list(getattr(re_assessment, "missing_information", []))

            trace.post_recovery_sufficiency = is_suff
            trace.post_recovery_probability = prob

            hop_trace = HopTrace(
                hop_number=hop,
                target_missing_facts=list(current_missing),
                formulated_query=reformulated_query,
                retrieved_chunk_ids=[c.chunk_id for c in retrieved_chunks],
                new_unique_chunk_ids=[c.chunk_id for c in new_chunks],
                sufficiency_after_hop=is_suff,
                probability_after_hop=prob,
                source_documents=hop_docs,
            )
            trace.hop_traces.append(hop_trace)

            if is_suff:
                trace.termination_reason = "SUFFICIENT_CONVERGENCE"
                break
            else:
                current_missing = new_missing if new_missing else current_missing
                if hop == effective_depth:
                    trace.termination_reason = "MAX_DEPTH_REACHED"

        # Construct final provenance chain
        trace.provenance_chain = [
            f"Hop-{ht.hop_number}: query='{ht.formulated_query}' -> docs={ht.source_documents} (+{len(ht.new_unique_chunk_ids)} chunks)"
            for ht in trace.hop_traces
        ]

        if last_assessment is None:
            last_assessment = judge_func(question, current_chunks)

        return current_chunks, last_assessment, trace
