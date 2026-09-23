"""Text chunking with document and page provenance."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_rag.document_loader import LoadedDocument


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    corpus_id: str
    document_id: str
    document_name: str
    page: int
    text: str


def chunk_documents(
    documents: list[LoadedDocument] | tuple[LoadedDocument, ...],
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> tuple[DocumentChunk, ...]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    chunks: list[DocumentChunk] = []

    for document in documents:
        text = " ".join(document.text.split())

        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}:chunk-{chunk_number}",
                        corpus_id=document.document_id,
                        document_id=document.document_id,
                        document_name=document.document_name,
                        page=document.page,
                        text=chunk_text,
                    )
                )

            if end >= len(text):
                break

            start = end - overlap
            chunk_number += 1

    return tuple(chunks)
