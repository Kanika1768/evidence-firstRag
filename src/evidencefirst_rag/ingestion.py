"""Document Ingestion Layer for EvidenceFirst RAG (Person 1 — Avni).

Extracts PDF, TXT, and Markdown files while preserving the file name and the
1-based page number of every page. Cleans running headers/footers that repeat
across pages, standalone page numbers, copyright banners, and blank text.
Detects PDF section headings from font size so the chunker can attach a
``section`` to every chunk.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

EDGE_LINES = 6
MIN_PAGES_FOR_REPEAT_DETECTION = 3
REPEAT_PAGE_FRACTION = 0.5
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_WORDS = 14


@dataclass(frozen=True)
class DocumentPage:
    """A single page of an ingested document with its provenance."""

    document_id: str
    document_name: str
    source_path: str
    page: int
    text: str
    headings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "page": self.page,
            "text": self.text,
            "headings": list(self.headings),
        }


def make_document_id(path: Path) -> str:
    """Stable, URL-safe document id derived from the file name (``NIST.AI.100-1.pdf`` -> ``nist-ai-100-1``)."""
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "document"


def clean_page_text(raw_text: str) -> str:
    """Normalize ligatures (NFKC) and remove standalone page numbers, copyright footers, and blank lines."""
    if not raw_text:
        return ""

    cleaned_lines: list[str] = []
    normalized = re.sub("\u00ad\\s*", "", unicodedata.normalize("NFKC", raw_text))
    for line in normalized.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            continue
        if re.match(r"^(?:page\s+)?\d+(?:\s*(?:of|\/)\s*\d+)?$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^[ivxlc]+$", stripped, re.IGNORECASE) and len(stripped) <= 6:
            continue
        if re.match(r"^(?:copyright|all rights reserved|\(c\))\b.*", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)
    return text.strip()


def _normalize_edge_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _page_number_keys(line: str, page_index: int) -> set[tuple[str, int]]:
    """(digit-masked line, number - page index) pairs; a running page number keeps the offset constant."""
    masked = re.sub(r"\d+", "#", _normalize_edge_line(line))
    return {(masked, int(n) - page_index) for n in re.findall(r"\d+", line)}


def remove_repeated_edge_lines(pages_lines: Sequence[Sequence[str]]) -> list[list[str]]:
    """Drop running headers/footers: lines at the top or bottom of a page that recur on many pages.

    Candidates are the first and last ``EDGE_LINES`` lines of each page; running lines
    are peeled off from each edge inward, so a repeated line inside the body is never removed.

    A line counts as running if the same text recurs, or if it differs only by a
    number that advances with the page (``Page 3`` / ``Page 4``, ``NIST AI 100-1 12``).
    Lines whose numbers vary independently of the page (``Table 4`` / ``Table 9``) are kept.
    """
    pages = [[line for line in lines if line.strip()] for lines in pages_lines]
    if len(pages) < MIN_PAGES_FOR_REPEAT_DETECTION:
        return pages

    exact: Counter[str] = Counter()
    numbered: Counter[tuple[str, int]] = Counter()
    for idx, lines in enumerate(pages):
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        exact.update({_normalize_edge_line(line) for line in edges})
        numbered.update(set().union(*(_page_number_keys(line, idx) for line in edges)) if edges else set())

    threshold = max(MIN_PAGES_FOR_REPEAT_DETECTION, int(len(pages) * REPEAT_PAGE_FRACTION + 0.5))
    repeated_exact = {line for line, count in exact.items() if count >= threshold}
    repeated_numbered = {key for key, count in numbered.items() if count >= threshold}

    def is_running(line: str, idx: int) -> bool:
        return _normalize_edge_line(line) in repeated_exact or bool(_page_number_keys(line, idx) & repeated_numbered)

    cleaned: list[list[str]] = []
    for idx, lines in enumerate(pages):
        start, end = 0, len(lines)
        while start < min(end, EDGE_LINES) and is_running(lines[start], idx):
            start += 1
        while end > max(start, len(lines) - EDGE_LINES) and is_running(lines[end - 1], idx):
            end -= 1
        cleaned.append(lines[start:end])
    return cleaned


def _is_heading_candidate(text: str) -> bool:
    words = text.split()
    return (
        0 < len(words) <= MAX_HEADING_WORDS
        and 3 <= len(text) <= 120
        and re.search(r"[A-Za-z]{2}", text) is not None
        and not text.endswith((".", ",", ";"))
        and not re.search(r"https?://|www\.", text)
    )


def _pymupdf_pages(path: Path) -> list[tuple[list[str], set[str]]]:
    """Return (lines, heading_lines) per page using PyMuPDF font information."""
    import pymupdf

    with pymupdf.open(path) as doc:
        raw_pages: list[list[tuple[str, float, bool]]] = []
        size_weights: Counter[float] = Counter()
        for page in doc:
            page_lines: list[tuple[str, float, bool]] = []
            for block in page.get_text("dict", sort=True)["blocks"]:
                for line in block.get("lines", []):
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
                    size = round(max(s["size"] for s in spans), 1)
                    bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
                    page_lines.append((text, size, bold))
                    for s in spans:
                        size_weights[round(s["size"], 1)] += len(s["text"])
            raw_pages.append(page_lines)

    body_size = size_weights.most_common(1)[0][0] if size_weights else 0.0
    result: list[tuple[list[str], set[str]]] = []
    for page_lines in raw_pages:
        headings = {
            text
            for text, size, bold in page_lines
            if _is_heading_candidate(text) and (size >= body_size * HEADING_SIZE_RATIO or (bold and size >= body_size))
        }
        result.append(([text for text, _, _ in page_lines], headings))
    return result


def _pypdf_pages(path: Path) -> list[tuple[list[str], set[str]]]:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return [((page.extract_text() or "").splitlines(), set()) for page in reader.pages]


def _raw_stream_text(path: Path) -> str:
    """Last-resort dependency-free extraction of literal text strings from compressed PDF streams.

    It cannot recover page boundaries, so everything is reported as page 1.
    """
    data = path.read_bytes()
    blocks: list[str] = []
    for stream in re.findall(rb"stream[\r\n]+([\s\S]*?)[\r\n]+endstream", data):
        try:
            decompressed = zlib.decompress(stream)
        except zlib.error:
            continue
        if decompressed.startswith((b"\x00\x01\x00\x00", b"OTTO", b"wOFF")):
            continue
        for bt in re.findall(rb"BT[\r\n]+([\s\S]*?)[\r\n]+ET", decompressed):
            words = []
            for literal in re.findall(rb"\((.*?)\)", bt):
                decoded = literal.decode("latin1", errors="ignore")
                if decoded and sum(c.isprintable() and ord(c) < 128 for c in decoded) / len(decoded) >= 0.8:
                    words.append(decoded)
            if words:
                blocks.append(" ".join(words).strip())
    return "\n".join(blocks)


def load_pdf(path: Path) -> list[DocumentPage]:
    """Extract PDF pages, preserving page numbers, removing running headers/footers, and detecting headings."""
    doc_id = make_document_id(path)
    extracted: list[tuple[list[str], set[str]]] = []
    for extractor in (_pymupdf_pages, _pypdf_pages):
        try:
            extracted = extractor(path)
        except ImportError:
            continue
        if any(lines for lines, _ in extracted):
            break

    if not any(lines for lines, _ in extracted):
        text = clean_page_text(_raw_stream_text(path))
        return [DocumentPage(doc_id, path.name, str(path), 1, text)] if text else []

    cleaned_lines = remove_repeated_edge_lines([lines for lines, _ in extracted])
    pages: list[DocumentPage] = []
    for page_num, (lines, (_, heading_lines)) in enumerate(zip(cleaned_lines, extracted), start=1):
        text = clean_page_text("\n".join(lines))
        if not text:
            continue
        kept_lines = set(text.splitlines())
        headings = tuple(dict.fromkeys(line for line in lines if line in heading_lines and line in kept_lines))
        pages.append(DocumentPage(doc_id, path.name, str(path), page_num, text, headings))
    return pages


def load_text(path: Path) -> list[DocumentPage]:
    """Ingest a plain-text or Markdown file as a single page."""
    cleaned = clean_page_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not cleaned:
        return []
    return [DocumentPage(make_document_id(path), path.name, str(path), 1, cleaned)]


def load_markdown(path: Path) -> list[DocumentPage]:
    """Ingest a Markdown file (``#`` headings are picked up by the chunker)."""
    return load_text(path)


def load_document(path: str | Path) -> list[DocumentPage]:
    """Load a single document based on its file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    if ext in {".txt", ".md"}:
        return load_text(p)
    raise ValueError(f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def load_corpus(directory: str | Path) -> list[DocumentPage]:
    """Recursively load all supported documents in a directory, in sorted path order."""
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(f"Directory not found: {d}")

    all_pages: list[DocumentPage] = []
    for file_path in sorted(d.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_pages.extend(load_document(file_path))
    return all_pages
