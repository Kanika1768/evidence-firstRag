"""Document Ingestion Layer for EvidenceFirst RAG (Person 1).

Extracts PDF, TXT, and Markdown files while preserving file names and page numbers.
Cleans repetitive running headers, running footers, and blank whitespace.
Attaches document_id, document_name, page, and source provenance.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    """Represents a single page or section of an ingested document."""

    document_id: str
    document_name: str
    source_path: str
    page: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "page": self.page,
            "text": self.text,
        }


def clean_page_text(raw_text: str) -> str:
    """Clean page text by stripping running headers, footers, and blank padding.

    - Strips standalone page numbers ("Page 1 of 12", "1 / 10", lone numbers)
    - Strips repetitive running header lines (e.g. copyright/module banners)
    - Collapses multiple blank lines while preserving paragraph boundaries
    """
    if not raw_text:
        return ""

    lines = raw_text.splitlines()
    cleaned_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Filter out standalone page number patterns
        if re.match(r"^(?:page\s+)?\d+(?:\s*(?:of|\/)\s*\d+)?$", stripped, re.IGNORECASE):
            continue
        # Filter out standalone copyright or generic running footers
        if re.match(r"^(?:copyright|all rights reserved|\(c\))\b.*", stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(stripped)

    # Join with single newline, preserving content paragraphs
    text = "\n".join(cleaned_lines)
    # Collapse 3+ consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def load_pdf(path: Path) -> list[DocumentPage]:
    """Extract PDF pages preserving page numbering and provenance."""
    pages: list[DocumentPage] = []
    doc_id = path.stem

    # Attempt 1: PyMuPDF (fitz)
    try:
        import pymupdf as fitz
        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc, start=1):
                raw = page.get_text("text")
                cleaned = clean_page_text(raw)
                if cleaned:
                    pages.append(
                        DocumentPage(
                            document_id=f"{doc_id}-p{page_num}",
                            document_name=path.name,
                            source_path=str(path),
                            page=page_num,
                            text=cleaned,
                        )
                    )
            if pages:
                return pages
    except (ImportError, Exception):
        pass

    # Attempt 2: pypdf / PyPDF2
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = clean_page_text(raw)
            if cleaned:
                pages.append(
                    DocumentPage(
                        document_id=f"{doc_id}-p{page_num}",
                        document_name=path.name,
                        source_path=str(path),
                        page=page_num,
                        text=cleaned,
                    )
                )
        if pages:
            return pages
    except (ImportError, Exception):
        pass

    # Attempt 3: Standalone pure-Python PDF stream decompressor
    try:
        data = path.read_bytes()
        stream_matches = re.findall(rb"stream[\r\n]+([\s\S]*?)[\r\n]+endstream", data)
        extracted_blocks: list[str] = []
        for s in stream_matches:
            try:
                decomp = zlib.decompress(s)
                # Skip font tables or image/binary blocks
                if (
                    decomp.startswith(b"\x00\x01\x00\x00")
                    or decomp.startswith(b"OTTO")
                    or decomp.startswith(b"wOFF")
                ):
                    continue

                # Look for PDF text objects BT ... ET
                bt_matches = re.findall(rb"BT[\r\n]+([\s\S]*?)[\r\n]+ET", decomp)
                for bt in bt_matches:
                    string_literals = re.findall(rb"\((.*?)\)", bt)
                    valid_words: list[str] = []
                    for st in string_literals:
                        decoded = st.decode("latin1", errors="ignore")
                        if not decoded:
                            continue
                        printable_ratio = sum(c.isprintable() and ord(c) < 128 for c in decoded) / len(decoded)
                        if printable_ratio >= 0.8:
                            valid_words.append(decoded)
                    if valid_words:
                        extracted_blocks.append(" ".join(valid_words).strip())
            except Exception:
                continue

        if extracted_blocks:
            full_text = "\n\n".join(extracted_blocks)
            cleaned = clean_page_text(full_text)
            if cleaned:
                pages.append(
                    DocumentPage(
                        document_id=f"{doc_id}-p1",
                        document_name=path.name,
                        source_path=str(path),
                        page=1,
                        text=cleaned,
                    )
                )
    except Exception:
        pass

    return pages


def load_text(path: Path) -> list[DocumentPage]:
    """Ingest plain text file as a page."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_page_text(text)
    if not cleaned:
        return []
    return [
        DocumentPage(
            document_id=path.stem,
            document_name=path.name,
            source_path=str(path),
            page=1,
            text=cleaned,
        )
    ]


def load_markdown(path: Path) -> list[DocumentPage]:
    """Ingest Markdown file."""
    return load_text(path)


def load_document(path: str | Path) -> list[DocumentPage]:
    """Load a single document based on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    elif ext in {".txt", ".md"}:
        return load_text(p)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def load_corpus(directory: str | Path) -> list[DocumentPage]:
    """Recursively load all supported documents in a directory."""
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")

    all_pages: list[DocumentPage] = []
    for file_path in sorted(d.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_pages.extend(load_document(file_path))

    return all_pages
