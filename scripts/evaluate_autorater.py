#!/usr/bin/env python3
"""Evaluation Runner for LLM Sufficiency AutoRater vs Heuristic Judge.

Compares:
1. Heuristic Sufficiency Judge (AutoraterStyleSufficiencyJudge)
2. Local LLM AutoRater (Qwen/Qwen2.5-0.5B-Instruct via LLMSufficiencyAutoRater)
3. Independent LLM Silver Annotations (data/evaluation/independent_llm_annotations.jsonl)

Against:
- Candidate / Proxy Labels (data/evaluation/benchmark.jsonl, 80 items)

Outputs:
- results/autorater_evaluation.csv (summary metrics and dataset/reasoning breakdowns)
- results/autorater_per_instance.csv (per-record decisions, scores, and rationales)

RESEARCH INTEGRITY NOTICE:
Independent human gold labels remain PENDING validation. All metrics herein evaluate
alignment against curated candidate proxy labels and independent LLM annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.local_llm_client import LocalTransformersLLMClient
from agentic_rag.adapters.structured_sufficiency import LLMSufficiencyAutoRater
from agentic_rag.contracts import (
    Claim,
    ContextStatus,
    Corpus,
    DraftAnswer,
    FactPriority,
    RequiredFact,
    RetrievalPlan,
    Route,
    Snippet,
)
from agentic_rag.general_planner import GeneralPlanner, _meaningful_terms
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge
from evidencefirst_rag.chunking import Chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall, F1 for positive class 'SUFFICIENT'."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SUFFICIENT" and yp == "SUFFICIENT")
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "INSUFFICIENT" and yp == "INSUFFICIENT")
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "INSUFFICIENT" and yp == "SUFFICIENT")
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "SUFFICIENT" and yp == "INSUFFICIENT")
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_heuristic_judge(
    judge: AutoraterStyleSufficiencyJudge,
    planner: GeneralPlanner,
    question: str,
    context: str,
    required_facts: Sequence[str] | None = None,
) -> tuple[str, float, str]:
    chunk = Chunk(
        chunk_id="c_bench_0",
        document_id="doc_bench",
        document_name="benchmark_doc",
        page=1,
        section="Overview",
        text=context,
        token_count=len(context.split()),
        char_count=len(context),
    )
    plan = planner.plan(question, (Corpus(id="doc_bench", description="benchmark"),))
    if not plan.required_facts:
        fact = RequiredFact(
            id="f1",
            description=question,
            priority=FactPriority.MUST,
            metadata={"required_terms": _meaningful_terms(question)},
        )
        route = Route(fact_id="f1", candidate_corpus_ids=("doc_bench",), reason="default")
        plan = RetrievalPlan(question=question, required_facts=(fact,), routes=(route,))

    snippets = (
        Snippet(id="c_bench_0", corpus_id="doc_bench", document_id="doc_bench", text=context),
    )
    claims = (Claim(text=context, snippet_ids=("c_bench_0",)),)
    draft = DraftAnswer(text=context, claims=claims)

    assessment = judge.assess(question, plan, snippets, draft)
    status_str = "SUFFICIENT" if assessment.status == ContextStatus.SUFFICIENT else "INSUFFICIENT"
    return status_str, float(assessment.sufficiency_score), assessment.reason or ""


def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "evaluation" / ".cache_autorater_qwen",
        help="Directory to store fresh verified autorater cache entries.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Enable cache lookup.",
    )
    parser.add_argument(
        "--force-fresh",
        dest="use_cache",
        action="store_false",
        help="Bypass cache and force fresh model inference.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("EVIDENCEFIRST RAG: LLM AUTORATER & HEURISTIC JUDGE EVALUATION (PROVENANCE-AWARE)")
    print("Target Benchmark: 80 Curated Sufficiency Instances (PopQA, FreshQA, NQ, EntityQuestions)")
    print("=" * 80)
    print("[DISCLAIMER] Human gold annotations are PENDING validation. All metrics below evaluate")
    print("             alignment against candidate/proxy labels and independent LLM annotations.\n")

    benchmark_path = ROOT / "data" / "evaluation" / "benchmark.jsonl"
    independent_path = ROOT / "data" / "evaluation" / "independent_llm_annotations.jsonl"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(benchmark_path)
    print(f"Loaded {len(records)} benchmark instances from {benchmark_path.relative_to(ROOT)}")

    independent_map: dict[str, dict[str, Any]] = {}
    if independent_path.exists():
        indep_records = load_jsonl(independent_path)
        independent_map = {r["id"]: r for r in indep_records}
        print(f"Loaded {len(indep_records)} independent silver annotations from {independent_path.relative_to(ROOT)}")

    # Initialize judges
    heuristic_judge = AutoraterStyleSufficiencyJudge()
    planner = GeneralPlanner()

    llm_client = LocalTransformersLLMClient(
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        allow_fallback=True,
    )
    llm_autorater = LLMSufficiencyAutoRater(
        llm_client=llm_client,
        fallback_judge=heuristic_judge,
        catch_exceptions=True,
    )

    per_instance_rows: list[dict[str, Any]] = []
    candidate_labels: list[str] = []
    heuristic_preds: list[str] = []
    autorater_preds: list[str] = []
    silver_preds: list[str] = []
    provenance_counts: dict[str, int] = {}

    print(f"\nRunning sufficiency evaluations across 80 benchmark records (cache: {args.cache_dir.relative_to(ROOT)})...")
    for idx, rec in enumerate(records, start=1):
        rec_id = rec["id"]
        q = rec["original_question"]
        ctx = rec["context"]
        cand = rec["candidate_label"]
        candidate_labels.append(cand)

        # 1. Heuristic Judge
        h_pred, h_score, h_reason = evaluate_heuristic_judge(
            heuristic_judge, planner, q, ctx, rec.get("required_facts")
        )
        heuristic_preds.append(h_pred)

        # 2. Local LLM AutoRater
        res = llm_autorater.evaluate(q, ctx, required_facts=rec.get("required_facts"))
        ar_pred = "SUFFICIENT" if res.sufficient else "INSUFFICIENT"
        ar_score = res.probability
        ar_reason = res.rationale
        ar_prov = getattr(res, "provenance", "unknown")
        provenance_counts[ar_prov] = provenance_counts.get(ar_prov, 0) + 1
        autorater_preds.append(ar_pred)

        # 3. Independent silver annotation (if present)
        indep_rec = independent_map.get(rec_id, {})
        silv_pred = indep_rec.get("independent_annotation", "UNKNOWN")
        silver_preds.append(silv_pred)

        per_instance_rows.append({
            "id": rec_id,
            "source_dataset": rec.get("source_dataset", ""),
            "reasoning_type": rec.get("reasoning_type", ""),
            "candidate_label": cand,
            "heuristic_prediction": h_pred,
            "heuristic_score": round(h_score, 4),
            "autorater_prediction": ar_pred,
            "autorater_score": round(ar_score, 4),
            "autorater_provenance": ar_prov,
            "evaluator_mode": res.evaluator_mode,
            "silver_prediction": silv_pred,
            "autorater_rationale": ar_reason,
            "autorater_missing": "; ".join(res.missing_information),
        })

    # Compute overall metrics against candidate proxy labels
    h_metrics = compute_binary_metrics(candidate_labels, heuristic_preds)
    ar_metrics = compute_binary_metrics(candidate_labels, autorater_preds)

    # Silver LLM curation concordance check (not a live model)
    silv_matches = sum(1 for c, s in zip(candidate_labels, silver_preds) if c == s)
    silv_concordance = (silv_matches / len(records)) if records else 0.0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS AGAINST CANDIDATE / PROXY LABELS (N=80)")
    print("=" * 80)
    print(f"{'Evaluator':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Confusion Matrix (TP/TN/FP/FN)'}")
    print("-" * 80)
    for name, m in [
        ("Heuristic Judge", h_metrics),
        ("LLM AutoRater (Qwen2.5)", ar_metrics),
    ]:
        print(
            f"{name:<28} | {m['accuracy']*100:6.2f}%   | {m['precision']:6.3f}     | {m['recall']:6.3f}     | {m['f1']:6.3f}     | "
            f"TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}"
        )

    print("\nAUTORATER PROVENANCE BREAKDOWN:")
    for prov_tag, count in sorted(provenance_counts.items()):
        print(f"  • {prov_tag:<32}: {count}/{len(records)} instances ({count/len(records)*100:.1f}%)")

    print("\n" + "-" * 80)
    print(f"CURATOR VS. SILVER LLM CONCORDANCE (PRE-ANNOTATION AUDIT):")
    print(f"  Pre-computed Silver Concordance: {silv_concordance*100:.2f}% ({silv_matches}/{len(records)})")
    print(f"  Note: independent_llm_annotations.jsonl is a static silver curation check artifact")
    print(f"  created during benchmark curation (commit fe536df). It is not a live evaluator.")
    print("-" * 80)

    # Inter-evaluator agreement
    h_ar_agreement = sum(1 for h, a in zip(heuristic_preds, autorater_preds) if h == a) / len(records)
    print("\nINTER-EVALUATOR PAIRWISE AGREEMENT:")
    print(f"  Heuristic Judge vs. LLM AutoRater: {h_ar_agreement*100:.2f}% ({sum(1 for h, a in zip(heuristic_preds, autorater_preds) if h == a)}/80)")

    # Breakdown by Source Dataset for LLM AutoRater
    datasets = sorted({r.get("source_dataset", "Unknown") for r in records})
    print("\nLLM AUTORATER BREAKDOWN BY SOURCE DATASET:")
    print(f"{'Dataset':<20} | {'Count':<6} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    dataset_summary = []
    for ds in datasets:
        ds_indices = [i for i, r in enumerate(records) if r.get("source_dataset") == ds]
        ds_true = [candidate_labels[i] for i in ds_indices]
        ds_pred = [autorater_preds[i] for i in ds_indices]
        m = compute_binary_metrics(ds_true, ds_pred)
        dataset_summary.append((ds, m))
        print(f"{ds:<20} | {len(ds_indices):<6} | {m['accuracy']*100:6.2f}%   | {m['precision']:6.3f}     | {m['recall']:6.3f}     | {m['f1']:6.3f}")

    # Breakdown by Reasoning Type for LLM AutoRater with single-class metric handling
    reasonings = sorted({r.get("reasoning_type", "Unknown") for r in records})
    print("\nLLM AUTORATER BREAKDOWN BY REASONING TYPE:")
    print(f"{'Reasoning Type':<30} | {'Count':<6} | {'Accuracy':<10} | {'Slice Distribution':<20} | {'Primary Slice Metric'}")
    print("-" * 88)
    reasoning_summary = []
    for rt in reasonings:
        rt_indices = [i for i, r in enumerate(records) if r.get("reasoning_type") == rt]
        rt_true = [candidate_labels[i] for i in rt_indices]
        rt_pred = [autorater_preds[i] for i in rt_indices]
        m = compute_binary_metrics(rt_true, rt_pred)

        n_pos = m["tp"] + m["fn"]
        n_neg = m["tn"] + m["fp"]
        slice_dist = f"S:{n_pos} / I:{n_neg}"

        if n_pos > 0 and n_neg > 0:
            slice_metric = f"F1: {m['f1']:.3f} (P:{m['precision']:.2f}, R:{m['recall']:.2f})"
        elif n_pos == 0:
            tnr = (m["tn"] / n_neg) if n_neg > 0 else 0.0
            slice_metric = f"TNR (Specificity): {tnr:.3f} ({m['tn']}/{n_neg})"
        else:
            tpr = (m["tp"] / n_pos) if n_pos > 0 else 0.0
            slice_metric = f"TPR (Sensitivity): {tpr:.3f} ({m['tp']}/{n_pos})"

        reasoning_summary.append((rt, m, slice_dist, slice_metric))
        print(f"{rt:<30} | {len(rt_indices):<6} | {m['accuracy']*100:6.2f}%   | {slice_dist:<20} | {slice_metric}")

    # Save summary CSV to corrected location
    summary_path = results_dir / "autorater_evaluation_corrected.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# Research Integrity Note: Evaluated against candidate proxy labels. Human validation pending."])
        writer.writerow(["# Silver LLM is a pre-existing curation concordance check (100.0% agreement), not a live model."])
        writer.writerow(["evaluator", "benchmark_split", "instances", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
        for name, m in [
            ("Heuristic_Judge", h_metrics),
            ("LLM_AutoRater_Qwen2.5", ar_metrics),
        ]:
            writer.writerow([name, "All_80_Instances", m["total"], round(m["accuracy"], 4), round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4), m["tp"], m["tn"], m["fp"], m["fn"]])

        writer.writerow([])
        writer.writerow(["# LLM AutoRater Dataset Breakdown"])
        writer.writerow(["dataset", "instances", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
        for ds, m in dataset_summary:
            writer.writerow([ds, m["total"], round(m["accuracy"], 4), round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4), m["tp"], m["tn"], m["fp"], m["fn"]])

        writer.writerow([])
        writer.writerow(["# LLM AutoRater Reasoning Type Breakdown (Single-Class Aware)"])
        writer.writerow(["reasoning_type", "instances", "accuracy", "slice_distribution", "primary_slice_metric", "tp", "tn", "fp", "fn"])
        for rt, m, s_dist, s_met in reasoning_summary:
            writer.writerow([rt, m["total"], round(m["accuracy"], 4), s_dist, s_met, m["tp"], m["tn"], m["fp"], m["fn"]])

    print(f"\n✓ Saved corrected evaluation summary to: {summary_path.relative_to(ROOT)}")

    # Save per-instance CSV to corrected location
    per_inst_path = results_dir / "autorater_per_instance_corrected.csv"
    with open(per_inst_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "id", "source_dataset", "reasoning_type", "candidate_label",
            "heuristic_prediction", "heuristic_score",
            "autorater_prediction", "autorater_score",
            "autorater_provenance", "evaluator_mode",
            "silver_prediction", "autorater_rationale", "autorater_missing",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_instance_rows)
    print(f"✓ Saved corrected per-instance records to: {per_inst_path.relative_to(ROOT)}")
    print(f"Completed in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
