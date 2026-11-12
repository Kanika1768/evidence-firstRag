#!/usr/bin/env python3
"""Evaluation Runner for Improved Retrieval on Difficult Queries.

Evaluates:
- Baselines: Dense-only (FAISS), BM25-only (Okapi), Hybrid (RRF Fusion), Hybrid + Reranker (Cross-Encoder)
- Improved: Adaptive Score Fusion (Dense+BM25), Adaptive Hybrid + Reranker (Improved)

On all 54 answerable queries in data/eval/retrieval_eval.jsonl.

Outputs:
- results/retrieval_ablation_improved.csv (aggregate table)
- results/retrieval_improved_per_query.csv (per-query rankings and metrics)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.retrieval import (
    EXTENDED_RETRIEVAL_MODES,
    MODE_LABELS,
    HybridRetriever,
    evaluate_retrieval_ablation,
    load_retrieval_eval_cases,
)


def run_improved_retrieval_evaluation(
    processed_dir: Path = ROOT / "data" / "processed",
    eval_file: Path = ROOT / "data" / "eval" / "retrieval_eval.jsonl",
    results_dir: Path = ROOT / "results",
) -> dict[str, dict[str, float]]:
    t_start = time.perf_counter()
    print("=" * 80)
    print("EVIDENCEFIRST RAG: RETRIEVAL IMPROVEMENT EVALUATION")
    print("Evaluating baseline vs. improved adaptive retrieval on NIST corpus")
    print("=" * 80)

    retriever = HybridRetriever.from_processed(processed_dir)
    cases = load_retrieval_eval_cases(eval_file, list(retriever.chunks_by_id.values()))
    print(f"Corpus Chunks : {len(retriever.chunks_by_id)}")
    print(f"Eval Questions: {len(cases)}")
    print(f"Dense Backend : {retriever.embedding_model.backend_name}")
    print(f"Reranker      : {retriever.reranker.backend_name}\n")

    out_csv = results_dir / "retrieval_ablation_improved.csv"
    per_query_csv = results_dir / "retrieval_improved_per_query.csv"

    results = evaluate_retrieval_ablation(
        cases,
        retriever,
        output_csv=out_csv,
        per_query_csv=per_query_csv,
        modes=EXTENDED_RETRIEVAL_MODES,
    )

    print(f"{'Configuration':40s} {'Recall@5':>10s} {'Recall@10':>10s} {'MRR':>10s} {'Latency (ms)':>14s}")
    print("-" * 88)
    for mode in EXTENDED_RETRIEVAL_MODES:
        m = results[mode]
        label = MODE_LABELS.get(mode, mode)
        print(f"{label:40s} {m['Recall@5']:10.4f} {m['Recall@10']:10.4f} {m['MRR']:10.4f} {m['median_latency_ms']:14.1f}")

    print("\n" + "=" * 80)
    print("DIFFICULT QUERY ABLATION SUMMARY:")
    base_rrf = results["hybrid"]
    base_ce = results["hybrid_rerank"]
    imp_adapt = results["adaptive_hybrid"]
    imp_ce = results["adaptive_hybrid_rerank"]

    mrr_gain_no_ce = (imp_adapt["MRR"] - base_rrf["MRR"]) / base_rrf["MRR"] * 100
    r5_gain_ce = (imp_ce["Recall@5"] - base_ce["Recall@5"]) / base_ce["Recall@5"] * 100
    r10_gain_ce = (imp_ce["Recall@10"] - base_ce["Recall@10"]) / base_ce["Recall@10"] * 100

    print(f"  • Adaptive Fusion vs. RRF (no reranker) : MRR improved by {mrr_gain_no_ce:+.2f}% ({base_rrf['MRR']:.4f} -> {imp_adapt['MRR']:.4f})")
    print(f"  • Improved Hybrid+Reranker vs. Baseline : Recall@5 gain of {r5_gain_ce:+.2f}% ({base_ce['Recall@5']:.4f} -> {imp_ce['Recall@5']:.4f})")
    print(f"  • Improved Hybrid+Reranker vs. Baseline : Recall@10 gain of {r10_gain_ce:+.2f}% ({base_ce['Recall@10']:.4f} -> {imp_ce['Recall@10']:.4f})")
    print(f"\n✓ Wrote: {out_csv.relative_to(ROOT)}")
    print(f"✓ Wrote: {per_query_csv.relative_to(ROOT)}")
    print(f"Completed in {time.perf_counter() - t_start:.2f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data" / "eval" / "retrieval_eval.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_improved_retrieval_evaluation(args.processed_dir, args.eval_file, args.results_dir)


if __name__ == "__main__":
    main()
