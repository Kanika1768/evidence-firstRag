"""General-purpose local document loading for EvidenceFirst RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pymupdf as fitz


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class LoadedDocument:
    document_id: str
    document_name: str
    source_path: str
    page: int
    text: str


def load_document(path: str | Path) -> Sequence[LoadedDocument]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type: {path.suffix}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if path.suffix.lower() == ".pdf":
        return _load_pdf(path)

    return _load_text(path)


def load_documents(directory: str | Path) -> Sequence[LoadedDocument]:
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")

    documents: list[LoadedDocument] = []

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_document(path))

    return tuple(documents)


def _load_pdf(path: Path) -> Sequence[LoadedDocument]:
    documents: list[LoadedDocument] = []

    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                LoadedDocument(
                    document_id=f"{path.stem}-p{page_number}",
                    document_name=path.name,
                    source_path=str(path),
                    page=page_number,
                    text=text,
                )
            )

    return tuple(documents)


def _load_text(path: Path) -> Sequence[LoadedDocument]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    if not text:
        return ()

    return (
        LoadedDocument(
            document_id=path.stem,
            document_name=path.name,
            source_path=str(path),
            page=1,
            text=text,
        ),
    )
