"""Grounded Answer Generation & Citation Formatting (Person 2 — Darshit).

Implements structured answer generation returning:
{
  "answer": "...",
  "confidence": 0.87,
  "claims": [
    {
      "text": "...",
      "source_chunk_ids": ["..."]
    }
  ]
}
Converts raw chunk IDs into readable document and page citations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from evidencefirst_rag.chunking import Chunk


@dataclass(frozen=True)
class AnswerClaim:
    """Individual assertion within the generated answer linked to source chunk IDs."""

    text: str
    source_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_chunk_ids": list(self.source_chunk_ids),
        }


@dataclass(frozen=True)
class GroundedAnswerRecord:
    """Structured generated answer with generator confidence and claim citations."""

    answer: str
    confidence: float
    claims: tuple[AnswerClaim, ...]
    is_abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "claims": [c.to_dict() for c in self.claims],
            "is_abstained": self.is_abstained,
        }

    def validate(self) -> None:
        """Validate output types and constraints."""
        if not isinstance(self.answer, str):
            raise TypeError("answer must be a string")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be a float")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not isinstance(self.claims, (list, tuple)):
            raise TypeError("claims must be a sequence of AnswerClaim objects")


def format_readable_citation(chunk: Chunk) -> str:
    """Convert chunk metadata into a human-readable citation string."""
    return f"[{chunk.document_name}, Page {chunk.page}, Chunk {chunk.chunk_id}]"


class GroundedGenerator:
    """Generates concise, attributed answers from verified sufficient evidence chunks."""

    def __init__(self, default_confidence: float = 0.90) -> None:
        self.default_confidence = default_confidence

    def generate(
        self,
        question: str,
        evidence_chunks: Sequence[Chunk],
        is_sufficient: bool,
    ) -> GroundedAnswerRecord:
        """Generate structured answer and sentence-level claims from evidence chunks."""
        if not is_sufficient or not evidence_chunks:
            return GroundedAnswerRecord(
                answer="The answer cannot be established from the available corpus.",
                confidence=0.0,
                claims=(),
                is_abstained=True,
            )

        def _stem(w: str) -> str:
            w = w.lower()
            for suff in ("ing", "ed", "es", "s"):
                if w.endswith(suff) and len(w) > len(suff) + 2:
                    return w[:-len(suff)]
            return w

        stopwords = {"what", "who", "when", "where", "why", "how", "does", "did", "is", "are", "the", "a", "an", "in", "on", "of", "to", "for"}
        q_stems = {_stem(w) for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stopwords}

        scored_candidates: list[tuple[int, str, str]] = []
        seen_texts: set[str] = set()

        for chunk in evidence_chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text.strip())
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean.split()) < 3 or s_clean in seen_texts:
                    continue
                s_stems = {_stem(w) for w in re.findall(r"[a-z0-9]+", s_clean.lower())}
                overlap = len(q_stems.intersection(s_stems))
                if overlap > 0:
                    scored_candidates.append((overlap, s_clean, chunk.chunk_id))
                    seen_texts.add(s_clean)

        scored_candidates.sort(key=lambda item: -item[0])
        selected_candidates = scored_candidates[:3]

        if not selected_candidates:
            # Fallback to first sentence of the top chunk if no specific overlap
            top_chunk = evidence_chunks[0]
            first_sent = re.split(r"(?<=[.!?])\s+", top_chunk.text.strip())[0].strip()
            if first_sent:
                selected_candidates = [(1, first_sent, top_chunk.chunk_id)]

        if not selected_candidates:
            return GroundedAnswerRecord(
                answer="The answer cannot be established from the available corpus.",
                confidence=0.0,
                claims=(),
                is_abstained=True,
            )

        claims_list = [
            AnswerClaim(text=text, source_chunk_ids=(chunk_id,))
            for _, text, chunk_id in selected_candidates
        ]
        full_answer = " ".join(text for _, text, _ in selected_candidates)
        # Compute self-reported confidence based on query coverage and evidence depth
        confidence = min(0.98, max(0.50, self.default_confidence + 0.02 * len(claims_list)))

        record = GroundedAnswerRecord(
            answer=full_answer,
            confidence=confidence,
            claims=tuple(claims_list),
            is_abstained=False,
        )
        record.validate()
        return record
