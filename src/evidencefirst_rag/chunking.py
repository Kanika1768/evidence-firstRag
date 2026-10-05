"""Document Chunking Layer for EvidenceFirst RAG (Person 1 — Avni).

Splits each page into windows of ``target_tokens`` tokens (default 550) where
consecutive windows share exactly ``overlap_tokens`` tokens (default 80).
Chunks never cross a page boundary, so every chunk has a single page number.
Each chunk carries full provenance: ``chunk_id``, ``document_id``,
``document_name``, ``page``, and ``section`` (the nearest preceding heading,
carried across pages of the same document).

Tokens are word runs and individual punctuation marks (``\\w+|[^\\w\\s]``), a
tokenizer-free approximation of sub-word token counts. Chunk text is sliced
from the original page text, so line breaks and headings are preserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from evidencefirst_rag.ingestion import DocumentPage

DEFAULT_TARGET_TOKENS = 550
DEFAULT_OVERLAP_TOKENS = 80

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


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
    """Approximate token count: word runs plus individual punctuation marks."""
    if not text:
        return 0
    return len(_TOKEN_RE.findall(text))


def _heading_from_line(line: str) -> str | None:
    """Return the heading text if a line is a Markdown heading or a short ALL-CAPS title of 2+ words."""
    line = line.strip()
    m_md = re.match(r"^#{1,6}\s+(.+)$", line)
    if m_md:
        return m_md.group(1).strip()
    if (
        re.match(r"^[A-Z0-9\s\-_&/]{3,60}:?$", line)
        and len(re.findall(r"\b[A-Z]{2,}\b", line)) >= 2
        and len(line.split()) <= 8
    ):
        return line.rstrip(":").strip()
    return None


def extract_section_heading(text: str, default_section: str = "General") -> str:
    """Return the first Markdown or ALL-CAPS heading in ``text``, else ``default_section``."""
    for line in text.splitlines():
        heading = _heading_from_line(line)
        if heading:
            return heading
    return default_section


def _heading_offsets(page: DocumentPage) -> list[tuple[int, str]]:
    """Character offsets of heading lines on a page (PDF font headings, Markdown, ALL-CAPS)."""
    known = set(page.headings)
    offsets: list[tuple[int, str]] = []
    pos = 0
    for line in page.text.split("\n"):
        stripped = line.strip()
        heading = stripped if stripped in known else _heading_from_line(stripped)
        if heading:
            offsets.append((pos, heading))
        pos += len(line) + 1
    return offsets


def chunk_page(
    page: DocumentPage,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    chunk_index_start: int = 0,
    carried_section: str | None = None,
) -> tuple[list[Chunk], int]:
    """Chunk one page into overlapping token windows.

    Returns the chunks and the next free chunk index for this document.
    ``carried_section`` is the last heading seen on earlier pages of the same
    document; it labels chunks that start before the first heading on this page.
    """
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    text = page.text
    spans = [m.span() for m in _TOKEN_RE.finditer(text)]
    if not spans:
        return [], chunk_index_start

    headings = _heading_offsets(page)
    default_section = carried_section or page.document_id
    step = target_tokens - overlap_tokens

    chunks: list[Chunk] = []
    idx = chunk_index_start
    for start in range(0, len(spans), step):
        end = min(len(spans), start + target_tokens)
        char_start, char_end = spans[start][0], spans[end - 1][1]
        preceding = [h for off, h in headings if off <= char_start]
        inside = [h for off, h in headings if char_start < off < char_end]
        section = preceding[-1] if preceding else (carried_section or (inside[0] if inside else default_section))
        chunk_text = text[char_start:char_end]
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:chunk-{idx}",
                document_id=page.document_id,
                document_name=page.document_name,
                page=page.page,
                section=section,
                text=chunk_text,
                token_count=end - start,
                char_count=len(chunk_text),
            )
        )
        idx += 1
        if end >= len(spans):
            break

    return chunks, idx


def chunk_document_pages(
    pages: Sequence[DocumentPage],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk pages of one or more documents.

    Chunk numbering restarts for every document, so a chunk id such as
    ``nist-ai-100-1:chunk-12`` stays stable when other documents are added.
    The current section carries over from one page to the next.
    """
    all_chunks: list[Chunk] = []
    next_index: dict[str, int] = {}
    last_section: dict[str, str] = {}

    for page in pages:
        page_chunks, next_index[page.document_id] = chunk_page(
            page,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            chunk_index_start=next_index.get(page.document_id, 0),
            carried_section=last_section.get(page.document_id),
        )
        page_headings = _heading_offsets(page)
        if page_headings:
            last_section[page.document_id] = page_headings[-1][1]
        all_chunks.extend(page_chunks)

    return all_chunks
