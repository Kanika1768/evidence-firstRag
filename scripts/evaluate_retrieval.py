#!/usr/bin/env python3
"""Retrieval ablation (Person 1 — Avni).

Compares dense-only, BM25-only, hybrid (RRF) and hybrid + cross-encoder reranker
on the questions in data/eval/retrieval_eval.jsonl, whose gold evidence is given
as verbatim quotes resolved to chunk ids in data/processed. Reports Recall@5,
Recall@10 and MRR.

Writes:
  results/retrieval_ablation.csv     one row per configuration
  results/retrieval_per_query.csv    per-question scores and top-5 chunk ids

Usage:
  python scripts/evaluate_retrieval.py [--processed-dir DIR] [--eval-file FILE]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.retrieval import (
    MODE_LABELS,
    RETRIEVAL_MODES,
    HybridRetriever,
    evaluate_retrieval_ablation,
    load_retrieval_eval_cases,
)


def run_retrieval_ablation(
    processed_dir: Path = ROOT / "data" / "processed",
    eval_file: Path = ROOT / "data" / "eval" / "retrieval_eval.jsonl",
    results_dir: Path = ROOT / "results",
) -> dict[str, dict[str, float]]:
    retriever = HybridRetriever.from_processed(processed_dir)
    cases = load_retrieval_eval_cases(eval_file, list(retriever.chunks_by_id.values()))
    print(f"Corpus: {len(retriever.chunks_by_id)} chunks | eval questions with evidence: {len(cases)}")
    print(f"Dense: {retriever.embedding_model.backend_name} ({retriever.vector_index.backend_name})")
    print(f"Reranker: {retriever.reranker.backend_name}")

    results = evaluate_retrieval_ablation(
        cases,
        retriever,
        output_csv=results_dir / "retrieval_ablation.csv",
        per_query_csv=results_dir / "retrieval_per_query.csv",
    )
    print(f"\n{'Configuration':38s} {'R@5':>7s} {'R@10':>7s} {'MRR':>7s} {'ms':>8s}")
    for mode in RETRIEVAL_MODES:
        m = results[mode]
        print(f"{MODE_LABELS[mode]:38s} {m['Recall@5']:7.4f} {m['Recall@10']:7.4f} {m['MRR']:7.4f} {m['median_latency_ms']:8.1f}")
    print(f"\nWrote {results_dir / 'retrieval_ablation.csv'} and {results_dir / 'retrieval_per_query.csv'}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data" / "eval" / "retrieval_eval.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_retrieval_ablation(args.processed_dir, args.eval_file, args.results_dir)


if __name__ == "__main__":
    main()
