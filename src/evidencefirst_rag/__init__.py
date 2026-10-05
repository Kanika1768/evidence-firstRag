"""EvidenceFirst RAG Package — Complete Team Implementation.

Architecture:
Documents -> Ingestion -> Chunking -> Dense + BM25 -> RRF Fusion -> Cross-Encoder
-> Top 6 Evidence Chunks -> Context Sufficiency Evaluator
-> Grounded Answer Draft -> Claim Verification -> Selective Abstention / Answer
-> UI Trace & Log
"""

from __future__ import annotations

__version__ = "1.0.0"
