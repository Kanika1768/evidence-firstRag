"""Claim-Level Citation Verification Layer (Person 2 — Darshit).

Checks whether material claims in the generated answer are entailed by their cited chunks.
If unsupported, the claim is flagged as ungrounded and prevents display of an untrusted answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.generation import AnswerClaim, GroundedAnswerRecord


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Verification outcome for an individual answer claim against cited evidence."""

    claim_text: str
    cited_chunk_ids: tuple[str, ...]
    is_supported: bool
    support_score: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_text": self.claim_text,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "is_supported": self.is_supported,
            "support_score": round(self.support_score, 4),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class VerificationReport:
    """Overall verification report across all claims in the generated answer."""

    all_supported: bool
    total_claims: int
    supported_count: int
    unsupported_count: int
    claim_results: tuple[ClaimVerificationResult, ...]
    decision: str  # 'TRUSTED', 'RECOVER', 'ABSTAIN'

    @property
    def overall_support_score(self) -> float:
        if not self.claim_results:
            return 1.0 if self.all_supported else 0.0
        return sum(r.support_score for r in self.claim_results) / len(self.claim_results)

    def to_dict(self) -> dict[str, object]:
        return {
            "all_supported": self.all_supported,
            "overall_support_score": round(self.overall_support_score, 4),
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "unsupported_count": self.unsupported_count,
            "claim_results": [r.to_dict() for r in self.claim_results],
            "decision": self.decision,
        }


class CitationVerifier:
    """Verifies that all generated answer claims are strictly grounded in cited chunks."""

    def __init__(self, min_support_threshold: float = 0.60) -> None:
        self.min_support_threshold = min_support_threshold

    def verify_claims(
        self,
        answer_record: GroundedAnswerRecord,
        chunks: Sequence[Chunk],
    ) -> VerificationReport:
        """Verify each claim against its cited chunk text."""
        if answer_record.is_abstained or not answer_record.claims:
            return VerificationReport(
                all_supported=True,
                total_claims=0,
                supported_count=0,
                unsupported_count=0,
                claim_results=(),
                decision="ABSTAIN",
            )

        chunks_by_id = {c.chunk_id: c for c in chunks}
        results: list[ClaimVerificationResult] = []

        for claim in answer_record.claims:
            claim_text = claim.text.strip()
            cited_ids = claim.source_chunk_ids

            # Find matching cited text
            cited_texts = [
                chunks_by_id[cid].text
                for cid in cited_ids
                if cid in chunks_by_id
            ]

            if not cited_texts:
                results.append(
                    ClaimVerificationResult(
                        claim_text=claim_text,
                        cited_chunk_ids=cited_ids,
                        is_supported=False,
                        support_score=0.0,
                        rationale="Cited chunk IDs do not exist in the retrieved evidence set.",
                    )
                )
                continue

            combined_cited_text = " ".join(cited_texts).lower()
            claim_tokens = [tok for tok in re.findall(r"[a-z0-9]+", claim_text.lower()) if len(tok) > 2]

            if not claim_tokens:
                score = 1.0
            else:
                matches = sum(1 for tok in claim_tokens if tok in combined_cited_text)
                score = matches / len(claim_tokens)

            is_supported = score >= self.min_support_threshold
            rationale = (
                f"Entailed by cited chunk ({int(score * 100)}% token support)."
                if is_supported
                else f"Insufficient evidence support ({int(score * 100)}% token support < {int(self.min_support_threshold * 100)}% threshold)."
            )

            results.append(
                ClaimVerificationResult(
                    claim_text=claim_text,
                    cited_chunk_ids=cited_ids,
                    is_supported=is_supported,
                    support_score=score,
                    rationale=rationale,
                )
            )

        supported_count = sum(1 for r in results if r.is_supported)
        unsupported_count = len(results) - supported_count
        all_supported = unsupported_count == 0

        decision = "TRUSTED" if all_supported else "ABSTAIN"

        return VerificationReport(
            all_supported=all_supported,
            total_claims=len(results),
            supported_count=supported_count,
            unsupported_count=unsupported_count,
            claim_results=tuple(results),
            decision=decision,
        )
