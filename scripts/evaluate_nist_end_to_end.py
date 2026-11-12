#!/usr/bin/env python3
"""End-to-End Comparative Evaluation on the Full 64-Query NIST Corpus.

Compares:
1. Baseline B0: Standard Naive RAG (Retrieve -> Generate, no sufficiency, no recovery, never abstains)
2. Baseline B1: Verification-Only RAG (Retrieve -> Generate -> Verify, no sufficiency check, no recovery)
3. EvidenceFirst E1: Full Agentic Pipeline (Hybrid Retrieve -> Sufficiency Judge -> Adaptive Recovery -> Grounded Generator -> Citation Verifier -> Selective Answering)

Corpus:
- 1,287 chunks across 24 NIST publications in data/processed/
- 64 benchmark queries (54 answerable + 10 unanswerable) in data/eval/retrieval_eval.jsonl

Metrics:
- Coverage / Answer Rate
- Groundedness (% verified claims)
- Unsupported Answer Rate (Hallucinations on unanswerable or insufficient context)
- Recovery Gain (queries recovered by adaptive retrieval)
- Accuracy on Answerable Set
- Median Latency (ms)

Outputs:
- results/nist_end_to_end_comparison.csv
- results/nist_per_query_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.generation import GroundedGenerator
from evidencefirst_rag.pipeline import EvidenceFirstPipeline
from evidencefirst_rag.recovery import AdaptiveRecoveryController
from evidencefirst_rag.retrieval import HybridRetriever
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator
from evidencefirst_rag.verifier import CitationVerifier


def load_nist_queries(eval_path: Path) -> list[dict[str, Any]]:
    queries = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def evaluate_b0(
    query_rec: dict[str, Any],
    retriever: HybridRetriever,
    generator: GroundedGenerator,
) -> dict[str, Any]:
    """Baseline B0: Naive RAG (never checks sufficiency, never abstains)."""
    t0 = time.perf_counter()
    q = query_rec["question"]
    chunks = retriever.retrieve(q, top_k=6, mode="hybrid_rerank")
    # Generates unconditionally assuming sufficiency
    ans = generator.generate(q, chunks, is_sufficient=True)
    latency = (time.perf_counter() - t0) * 1000.0

    is_answerable = query_rec.get("answerable", True)
    # B0 answers everything; unsupported if question is unanswerable or no chunks
    unsupported = not is_answerable or len(chunks) == 0

    return {
        "qid": query_rec["qid"],
        "system": "B0_Standard_RAG",
        "decision": "ANSWERED",
        "answer": ans.answer,
        "is_abstained": False,
        "unsupported": unsupported,
        "latency_ms": round(latency, 2),
    }


def evaluate_b1(
    query_rec: dict[str, Any],
    retriever: HybridRetriever,
    generator: GroundedGenerator,
    verifier: CitationVerifier,
) -> dict[str, Any]:
    """Baseline B1: Verify-Only RAG (no sufficiency judge, no recovery)."""
    t0 = time.perf_counter()
    q = query_rec["question"]
    chunks = retriever.retrieve(q, top_k=6, mode="hybrid_rerank")
    ans = generator.generate(q, chunks, is_sufficient=True)
    report = verifier.verify_claims(ans, chunks)
    latency = (time.perf_counter() - t0) * 1000.0

    # Abstains only if citations are completely unsupported
    is_abstained = not report.all_supported or ans.is_abstained
    decision = "ABSTAINED" if is_abstained else "ANSWERED"
    is_answerable = query_rec.get("answerable", True)
    unsupported = (decision == "ANSWERED") and not is_answerable

    return {
        "qid": query_rec["qid"],
        "system": "B1_Verify_Only_RAG",
        "decision": decision,
        "answer": ans.answer if not is_abstained else "The answer cannot be established from the available corpus.",
        "is_abstained": is_abstained,
        "unsupported": unsupported,
        "latency_ms": round(latency, 2),
    }


def evaluate_e1(
    query_rec: dict[str, Any],
    pipeline: EvidenceFirstPipeline,
) -> dict[str, Any]:
    """EvidenceFirst E1: Full pipeline with Sufficiency + Recovery + Verifier."""
    t0 = time.perf_counter()
    q = query_rec["question"]
    trace = pipeline.run(q)
    latency = (time.perf_counter() - t0) * 1000.0

    is_answerable = query_rec.get("answerable", True)
    is_abstained = trace.final_decision == "ABSTAINED"
    unsupported = (trace.final_decision == "ANSWERED") and not is_answerable
    recovery_triggered = bool(trace.recovery_trace and trace.recovery_trace.triggered)
    recovery_success = bool(trace.recovery_trace and trace.recovery_trace.post_recovery_sufficiency)

    return {
        "qid": query_rec["qid"],
        "system": "E1_EvidenceFirst_RAG",
        "decision": trace.final_decision,
        "answer": trace.final_answer,
        "is_abstained": is_abstained,
        "unsupported": unsupported,
        "recovery_triggered": recovery_triggered,
        "recovery_success": recovery_success,
        "latency_ms": round(latency, 2),
    }


def run_nist_end_to_end_evaluation(
    processed_dir: Path = ROOT / "data" / "processed",
    eval_file: Path = ROOT / "data" / "eval" / "retrieval_eval.jsonl",
    results_dir: Path = ROOT / "results",
) -> None:
    t_start = time.perf_counter()
    print("=" * 80)
    print("EVIDENCEFIRST RAG: FULL 64-QUERY NIST END-TO-END COMPARATIVE EVALUATION")
    print("Evaluating B0 (Standard RAG) vs B1 (Verify-Only) vs E1 (EvidenceFirst)")
    print("=" * 80)

    queries = load_nist_queries(eval_file)
    answerable_count = sum(1 for q in queries if q.get("answerable", True))
    unanswerable_count = len(queries) - answerable_count
    print(f"Total Benchmark Queries: {len(queries)} ({answerable_count} Answerable, {unanswerable_count} Unanswerable)")

    print(f"Loading NIST corpus from {processed_dir.relative_to(ROOT)}...")
    retriever = HybridRetriever.from_processed(processed_dir)
    print(f"Corpus Loaded: {len(retriever.chunks_by_id)} chunks across 24 NIST PDF publications.")

    evaluator = ContextSufficiencyEvaluator()
    recovery = AdaptiveRecoveryController(max_recovery_attempts=2)
    generator = GroundedGenerator(sentence_scorer=retriever.reranker.sentence_scorer)
    verifier = CitationVerifier()

    pipeline_e1 = EvidenceFirstPipeline(
        chunks=list(retriever.chunks_by_id.values()),
        retriever=retriever,
        evaluator=evaluator,
        recovery_controller=recovery,
        generator=generator,
        verifier=verifier,
    )

    per_query_rows: list[dict[str, Any]] = []
    b0_results, b1_results, e1_results = [], [], []

    print("\nRunning comparative pipeline evaluation across all 64 queries...")
    for idx, qrec in enumerate(queries, start=1):
        qid = qrec["qid"]
        is_ans = qrec.get("answerable", True)

        res_b0 = evaluate_b0(qrec, retriever, generator)
        res_b1 = evaluate_b1(qrec, retriever, generator, verifier)
        res_e1 = evaluate_e1(qrec, pipeline_e1)

        b0_results.append(res_b0)
        b1_results.append(res_b1)
        e1_results.append(res_e1)

        per_query_rows.append({
            "qid": qid,
            "question": qrec["question"],
            "answerable": is_ans,
            "b0_decision": res_b0["decision"],
            "b0_unsupported": res_b0["unsupported"],
            "b0_latency_ms": res_b0["latency_ms"],
            "b1_decision": res_b1["decision"],
            "b1_unsupported": res_b1["unsupported"],
            "b1_latency_ms": res_b1["latency_ms"],
            "e1_decision": res_e1["decision"],
            "e1_unsupported": res_e1["unsupported"],
            "e1_recovery_triggered": res_e1["recovery_triggered"],
            "e1_recovery_success": res_e1["recovery_success"],
            "e1_latency_ms": res_e1["latency_ms"],
        })

    # Summary metrics computation
    def aggregate_metrics(system_name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(results)
        answered = sum(1 for r in results if r["decision"] == "ANSWERED")
        abstained = sum(1 for r in results if r["decision"] == "ABSTAINED")
        unsupported = sum(1 for r in results if r["unsupported"])
        latencies = [r["latency_ms"] for r in results]

        # On unanswerable subset
        unans_results = [r for i, r in enumerate(results) if not queries[i].get("answerable", True)]
        unans_abstained = sum(1 for r in unans_results if r["decision"] == "ABSTAINED")
        unans_rate = (unans_abstained / len(unans_results)) if unans_results else 0.0

        # On answerable subset
        ans_results = [r for i, r in enumerate(results) if queries[i].get("answerable", True)]
        ans_answered = sum(1 for r in ans_results if r["decision"] == "ANSWERED")
        ans_rate = (ans_answered / len(ans_results)) if ans_results else 0.0

        return {
            "system": system_name,
            "total_queries": total,
            "coverage_answered": round(answered / total, 4),
            "abstention_rate": round(abstained / total, 4),
            "unsupported_total_rate": round(unsupported / total, 4),
            "unsupported_answered_rate": round((unsupported / answered) if answered > 0 else 0.0, 4),
            "answerable_coverage": round(ans_rate, 4),
            "unanswerable_abstention_accuracy": round(unans_rate, 4),
            "median_latency_ms": round(statistics.median(latencies), 1),
        }

    summary_b0 = aggregate_metrics("B0_Standard_RAG", b0_results)
    summary_b1 = aggregate_metrics("B1_Verify_Only_RAG", b1_results)
    summary_e1 = aggregate_metrics("E1_EvidenceFirst_RAG", e1_results)

    e1_recovered = sum(1 for r in e1_results if r.get("recovery_success", False))

    print("\n" + "=" * 105)
    print("NIST 64-QUERY END-TO-END PIPELINE COMPARISON RESULTS (CORRECTED)")
    print("=" * 105)
    print(f"{'System':<22} | {'Coverage':<10} | {'Ans Coverage':<13} | {'Unsupp/Tot':<11} | {'Unsupp/Ans':<11} | {'Unans Abstain':<13} | {'Med Lat(ms)':<12}")
    print("-" * 105)
    for s in [summary_b0, summary_b1, summary_e1]:
        print(
            f"{s['system']:<22} | {s['coverage_answered']*100:6.1f}%    | {s['answerable_coverage']*100:6.1f}%       | "
            f"{s['unsupported_total_rate']*100:6.1f}%     | {s['unsupported_answered_rate']*100:6.1f}%     | "
            f"{s['unanswerable_abstention_accuracy']*100:6.1f}%        | {s['median_latency_ms']:10.1f}"
        )

    print("\nEVIDENCEFIRST ADAPTIVE RECOVERY HIGHLIGHTS ON NIST:")
    print(f"  • Queries Triggering Recovery : {sum(1 for r in e1_results if r['recovery_triggered'])}/64")
    print(f"  • Queries Successfully Rescued: {e1_recovered}")
    print(f"  • Unsupported Hallucination Reduction: {summary_b0['unsupported_total_rate']*100:.1f}% (B0) -> {summary_e1['unsupported_total_rate']*100:.1f}% (E1)")

    # Save summary CSV to corrected destination
    comp_csv = results_dir / "nist_end_to_end_comparison_corrected.csv"
    with open(comp_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "system", "total_queries", "coverage_answered", "abstention_rate",
            "unsupported_total_rate", "unsupported_answered_rate", "answerable_coverage",
            "unanswerable_abstention_accuracy", "median_latency_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([summary_b0, summary_b1, summary_e1])
    print(f"\n✓ Saved comparative aggregate table to: {comp_csv.relative_to(ROOT)}")

    # Save per-query CSV to corrected destination
    per_q_csv = results_dir / "nist_per_query_comparison_corrected.csv"
    with open(per_q_csv, "w", newline="", encoding="utf-8") as f:
        fields = list(per_query_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_query_rows)
    print(f"✓ Saved per-query comparisons to: {per_q_csv.relative_to(ROOT)}")
    print(f"Completed in {time.perf_counter() - t_start:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data" / "eval" / "retrieval_eval.jsonl")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run_nist_end_to_end_evaluation(args.processed_dir, args.eval_file, args.results_dir)


if __name__ == "__main__":
    main()
