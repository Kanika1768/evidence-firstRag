#!/usr/bin/env python3
"""Corpus ingestion & indexing pipeline (Person 1 — Avni).

Loads every PDF/TXT/MD file under the input directories (default: the NIST demo
corpus in data/corpus/pdfs, fetched with scripts/download_corpus.py), removes
running headers/footers, chunks pages into 550-token windows with 80-token
overlap, and writes to data/processed/:

  chunks.parquet        chunk text + provenance (needs pandas/pyarrow)
  chunks.jsonl          the same rows, readable without extra dependencies
  faiss.index           dense index (native FAISS IndexFlatIP) + faiss.index.ids.json
  manifest.json         frozen config: inputs + sha256, chunking params, model backends

Usage:
  python scripts/ingest.py [--input-dir DIR ...] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.chunking import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, chunk_document_pages
from evidencefirst_rag.ingestion import SUPPORTED_EXTENSIONS, load_corpus
from evidencefirst_rag.retrieval import (
    ENCODER_WINDOW_OVERLAP,
    ENCODER_WINDOW_TOKENS,
    RRF_K,
    TOP_K_EVIDENCE,
    CrossEncoderReranker,
    DenseEmbeddingModel,
    build_vector_index,
)


def run_ingest(
    input_dirs: list[Path],
    output_dir: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> dict[str, object]:
    started = time.perf_counter()
    pages = []
    files = []
    for d in input_dirs:
        if not d.is_dir():
            raise SystemExit(f"Input directory not found: {d} (run scripts/download_corpus.py first?)")
        dir_pages = load_corpus(d)
        print(f"Loaded {len(dir_pages)} pages from {d}")
        pages.extend(dir_pages)
        files.extend(p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not pages:
        raise SystemExit("No documents found to ingest.")

    chunks = chunk_document_pages(pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise SystemExit("Duplicate chunk ids produced; refusing to write an inconsistent index.")
    print(f"Chunked into {len(chunks)} chunks ({target_tokens} tokens, {overlap_tokens} overlap)")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [c.to_dict() for c in chunks]
    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output_dir / "chunks.parquet", index=False)
        print(f"Wrote {output_dir / 'chunks.parquet'}")
    except ImportError:
        print("pandas/pyarrow not installed: skipped chunks.parquet (chunks.jsonl written)")

    embedder = DenseEmbeddingModel()
    print(f"Embedding with {embedder.backend_name} ...")
    index = build_vector_index(chunks, embedder)
    index.save(output_dir / "faiss.index")
    print(f"Wrote {output_dir / 'faiss.index'} ({len(index.chunk_ids)} window vectors for {len(chunks)} chunks, {index.backend_name})")

    token_counts = sorted(c.token_count for c in chunks)
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": [
            {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files
        ],
        "documents": len({c.document_id for c in chunks}),
        "pages": len(pages),
        "chunks": len(chunks),
        "chunk_tokens": {"min": token_counts[0], "median": token_counts[len(token_counts) // 2], "max": token_counts[-1]},
        "chunking": {"target_tokens": target_tokens, "overlap_tokens": overlap_tokens, "tokenizer": r"regex \w+|[^\w\s]"},
        "embedding_backend": embedder.backend_name,
        "embedding_dim": embedder.dim,
        "encoder_windows": {"tokens": ENCODER_WINDOW_TOKENS, "overlap": ENCODER_WINDOW_OVERLAP},
        "dense_vectors": len(index.chunk_ids),
        "vector_index_backend": index.backend_name,
        "bm25": {"k1": 1.5, "b": 0.75, "persisted": False, "note": "rebuilt from chunks at load time"},
        "rrf_k": RRF_K,
        "reranker_backend": CrossEncoderReranker().backend_name,
        "top_k_evidence": TOP_K_EVIDENCE,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'manifest.json'} in {time.perf_counter() - started:.1f}s")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, action="append", help="Directory of documents (repeatable).")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()
    run_ingest(
        input_dirs=args.input_dir or [ROOT / "data" / "corpus" / "pdfs"],
        output_dir=args.output_dir,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )


if __name__ == "__main__":
    main()
