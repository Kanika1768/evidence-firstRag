"""Document Chunking Layer for EvidenceFirst RAG (Person 1).

Implements ~550-token chunking with ~80-token overlap, preserving headings and
full document provenance: document_id, document_name, page, section, chunk_id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from evidencefirst_rag.ingestion import DocumentPage


@dataclass(frozen=True)
class Chunk:
    """A granular retrieval unit with full document and page provenance."""

    chunk_id: str
    document_id: str
    document_name: str
    page: int
    section: str
    text: str
    token_count: int
    char_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page": self.page,
            "section": self.section,
            "text": self.text,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            document_id=str(data["document_id"]),
            document_name=str(data["document_name"]),
            page=int(data["page"]),
            section=str(data.get("section", "General")),
            text=str(data["text"]),
            token_count=int(data.get("token_count", 0)),
            char_count=int(data.get("char_count", len(str(data["text"])))),
        )


def count_tokens(text: str) -> int:
    """Approximate token count using standard word/punctuation tokenization."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def extract_section_heading(text: str, default_section: str = "General") -> str:
    """Detect section header from Markdown (# Header) or capitalized title lines."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Markdown heading (# Section Title)
        m_md = re.match(r"^#{1,4}\s+(.+)$", line)
        if m_md:
            return m_md.group(1).strip()
        # All-caps or title header with colon (e.g. "SECTION 1: OVERVIEW")
        if re.match(r"^[A-Z0-9\s\-_]{3,40}(?::|$)", line) and len(line.split()) <= 6:
            return line.rstrip(":").strip()
    return default_section


def chunk_page(
    page: DocumentPage,
    target_tokens: int = 550,
    overlap_tokens: int = 80,
    chunk_index_start: int = 0,
) -> tuple[list[Chunk], int]:
    """Chunk a single DocumentPage into sliding window chunks of target_tokens with overlap_tokens."""
    text = page.text.strip()
    if not text:
        return [], chunk_index_start

    words = re.findall(r"\S+", text)
    total_words = len(words)

    # If document is within target size, yield a single chunk
    if total_words <= target_tokens:
        section = extract_section_heading(text, default_section=page.document_id)
        chunk = Chunk(
            chunk_id=f"{page.document_id}:chunk-{chunk_index_start}",
            document_id=page.document_id,
            document_name=page.document_name,
            page=page.page,
            section=section,
            text=text,
            token_count=count_tokens(text),
            char_count=len(text),
        )
        return [chunk], chunk_index_start + 1

    chunks: list[Chunk] = []
    step = max(1, target_tokens - overlap_tokens)
    current_idx = chunk_index_start
    section = extract_section_heading(text, default_section=page.document_id)

    for start_idx in range(0, total_words, step):
        end_idx = min(total_words, start_idx + target_tokens)
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words).strip()

        # Update heading context if present in this slice
        slice_section = extract_section_heading(chunk_text, default_section=section)
        section = slice_section

        chunk = Chunk(
            chunk_id=f"{page.document_id}:chunk-{current_idx}",
            document_id=page.document_id,
            document_name=page.document_name,
            page=page.page,
            section=section,
            text=chunk_text,
            token_count=count_tokens(chunk_text),
            char_count=len(chunk_text),
        )
        chunks.append(chunk)
        current_idx += 1

        if end_idx >= total_words:
            break

    return chunks, current_idx


def chunk_document_pages(
    pages: Sequence[DocumentPage],
    target_tokens: int = 550,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """Chunk a sequence of DocumentPages across an entire document or corpus."""
    all_chunks: list[Chunk] = []
    chunk_counter = 0

    for page in pages:
        page_chunks, chunk_counter = chunk_page(
            page,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            chunk_index_start=chunk_counter,
        )
        all_chunks.extend(page_chunks)

    return all_chunks
