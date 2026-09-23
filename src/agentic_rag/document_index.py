"""Build a searchable local document index for EvidenceFirst RAG."""

from __future__ import annotations

from typing import Sequence
from pathlib import Path
from agentic_rag.chunking import DocumentChunk, chunk_documents
from agentic_rag.document_loader import LoadedDocument, load_documents
from agentic_rag.adapters.retriever import LexicalDocument, LexicalRetriever


def build_local_index(
    documents: Sequence[LoadedDocument],
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    per_query_limit: int = 5,
) -> LexicalRetriever:
    chunks = chunk_documents(
        tuple(documents),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    lexical_documents = tuple(
        _to_lexical_document(chunk)
        for chunk in chunks
    )

    return LexicalRetriever(
        lexical_documents,
        per_query_limit=per_query_limit,
    )


def build_local_index_from_directory(
    directory: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
    per_query_limit: int = 5,
) -> LexicalRetriever:
    documents = load_documents(directory)

    return build_local_index(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
        per_query_limit=per_query_limit,
    )


def _to_lexical_document(chunk: DocumentChunk) -> LexicalDocument:
    return LexicalDocument(
        corpus_id=Path(chunk.document_name).stem,
        document_id=chunk.chunk_id,
        text=chunk.text,
        metadata={
            "document_name": chunk.document_name,
            "page": chunk.page,
        },
    )
