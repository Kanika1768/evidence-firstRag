#!/usr/bin/env python3
"""Comparative Evaluation Runner for ICLR 2025 Paper-Aligned AutoRater.

Evaluates 4 configurations on the 80-item Sufficient Context Benchmark:
1. Heuristic Baseline: Deterministic AutoraterStyleSufficiencyJudge
2. Qwen Zero-Shot Baseline: Qwen2.5-0.5B-Instruct using legacy sufficiency_v1 zero-shot prompt
3. Qwen ICLR 2025 One-Shot: Qwen2.5-0.5B-Instruct using faithful 1-shot CoT prompt
4. Gemini ICLR 2025 One-Shot: Gemini (gemini-3.5-flash-lite) using faithful 1-shot CoT prompt

RESEARCH INTEGRITY DISCLOSURE:
- Evaluates against candidate proxy labels (80 records: 40 SUFFICIENT, 40 INSUFFICIENT).
- Human gold labels remain strictly PENDING.
- Zero oracle leakage: Planned required facts are completely excluded from the ICLR 2025 prompts.
- All new evaluation outputs are saved to dedicated iclr2025_* files, preserving existing CSVs.

Outputs:
- results/iclr2025_autorater_benchmark.csv
- results/iclr2025_autorater_per_instance.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.llm_client import GenericRestLLMClient, create_llm_client, get_llm_credentials
from agentic_rag.adapters.local_llm_client import LocalTransformersLLMClient
from agentic_rag.adapters.structured_sufficiency import LLMSufficiencyAutoRater, StructuredSufficiencyResult
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
from agentic_rag.prompts.sufficiency_iclr2025 import ICLR2025_PROMPT_VERSION
from agentic_rag.prompts.sufficiency_v1 import SUFFICIENCY_PROMPT_VERSION
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge
from evidencefirst_rag.chunking import Chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_iclr2025")


class CachingGeminiClient:
    """Wrapper around GenericRestLLMClient with disk caching and provenance tracking."""

    def __init__(self, inner_client: GenericRestLLMClient, cache_dir: Path) -> None:
        self.inner_client = inner_client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.last_provenance: str = ""

    def generate_with_provenance(self, prompt: str) -> tuple[str, str]:
        key = hashlib.sha256(f"{self.inner_client.model}:{prompt.strip()}".encode("utf-8")).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    prov = data.get("provenance", "gemini_cache_hit")
                    if not prov.endswith("_cache_hit"):
                        prov = f"{prov}_cache_hit"
                    self.last_provenance = prov
                    return data["response"], prov
            except Exception:
                pass

        # Live invocation
        response = self.inner_client.complete(prompt)
        prov = f"gemini_live_api_{self.inner_client.model}"
        record = {
            "model": self.inner_client.model,
            "prompt": prompt,
            "response": response,
            "provenance": prov,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to write cache: %s", exc)

        self.last_provenance = prov
        return response, prov


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_binary_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
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


def evaluate_heuristic(
    judge: AutoraterStyleSufficiencyJudge,
    planner: GeneralPlanner,
    question: str,
    context: str,
) -> tuple[str, float, str]:
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
        "--benchmark-file",
        type=Path,
        default=ROOT / "data" / "evaluation" / "benchmark.jsonl",
        help="Path to 80-item benchmark JSONL file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory to save output evaluation CSVs.",
    )
    parser.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Skip Gemini API evaluation (e.g. if offline).",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip local Qwen evaluation.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVIDENCEFIRST RAG: ICLR 2025 SUFFICIENT CONTEXT AUTORATER BENCHMARK")
    print("=" * 80)

    records = load_benchmark(args.benchmark_file)
    print(f"Loaded {len(records)} benchmark instances from {args.benchmark_file.relative_to(ROOT)}")
    candidate_labels = [r["candidate_label"] for r in records]
    print(f"Candidate Proxy Labels: {candidate_labels.count('SUFFICIENT')} SUFFICIENT, {candidate_labels.count('INSUFFICIENT')} INSUFFICIENT")

    # 1. Setup Heuristic Baseline
    h_judge = AutoraterStyleSufficiencyJudge()
    planner = GeneralPlanner()

    # 2. Setup Qwen Zero-Shot Baseline (Legacy Prompt v1.0)
    qwen_zero_client = None
    qwen_zero_autorater = None
    if not args.skip_qwen:
        qwen_zero_client = LocalTransformersLLMClient(
            cache_dir=ROOT / "data" / "evaluation" / ".cache_autorater_qwen",
            use_cache=True,
            max_new_tokens=160,
        )
        qwen_zero_autorater = LLMSufficiencyAutoRater(
            llm_client=qwen_zero_client,
            prompt_version=SUFFICIENCY_PROMPT_VERSION,
            catch_exceptions=True,
        )

    # 3. Setup Qwen ICLR 2025 One-Shot (New 1-shot CoT Prompt)
    qwen_oneshot_client = None
    qwen_oneshot_autorater = None
    if not args.skip_qwen:
        qwen_oneshot_client = LocalTransformersLLMClient(
            cache_dir=ROOT / "data" / "evaluation" / ".cache_autorater_qwen_iclr2025",
            use_cache=True,
            max_new_tokens=512,
        )
        qwen_oneshot_autorater = LLMSufficiencyAutoRater(
            llm_client=qwen_oneshot_client,
            prompt_version=ICLR2025_PROMPT_VERSION,
            catch_exceptions=True,
        )

    # 4. Setup Gemini ICLR 2025 One-Shot
    gemini_autorater = None
    provider, api_key, model = get_llm_credentials()
    if not args.skip_gemini and provider == "gemini" and api_key:
        raw_gemini_client = GenericRestLLMClient(
            provider="gemini",
            api_key=api_key,
            model=model or "gemini-3.5-flash-lite",
            response_mime_type="text/plain",
            timeout=25.0,
        )
        caching_gemini = CachingGeminiClient(
            inner_client=raw_gemini_client,
            cache_dir=ROOT / "data" / "evaluation" / ".cache_autorater_gemini_iclr2025",
        )
        gemini_autorater = LLMSufficiencyAutoRater(
            llm_client=caching_gemini,
            prompt_version=ICLR2025_PROMPT_VERSION,
            catch_exceptions=True,
        )
        print(f"Gemini API Configured: {model or 'gemini-3.5-flash-lite'} (Caching in .cache_autorater_gemini_iclr2025)")
    else:
        if args.skip_gemini:
            print("Gemini API: Skipped via --skip-gemini")
        else:
            print("Gemini API: No credentials detected. Skipping Gemini.")

    # Run evaluations
    print("\nRunning benchmark across all configurations...")
    per_instance_rows: list[dict[str, Any]] = []

    heuristic_preds: list[str] = []
    qwen_zero_preds: list[str] = []
    qwen_oneshot_preds: list[str] = []
    gemini_preds: list[str] = []

    for idx, rec in enumerate(records, start=1):
        qid = rec["id"]
        q = rec["original_question"]
        ctx = rec["context"]
        cand = rec["candidate_label"]
        req_facts = rec.get("required_facts", [])

        # 1. Heuristic
        h_pred, h_score, h_reason = evaluate_heuristic(h_judge, planner, q, ctx)
        heuristic_preds.append(h_pred)

        # 2. Qwen Zero-shot (Legacy)
        qz_pred, qz_score, qz_prov, qz_rat = "SKIPPED", 0.0, "skipped", ""
        if qwen_zero_autorater is not None:
            # Note: Legacy prompt v1.0 included required_facts
            chunk = Chunk("c0", "d0", "d0", 1, "s0", ctx, len(ctx.split()), len(ctx))
            res_qz = qwen_zero_autorater.evaluate(q, [chunk], required_facts=req_facts)
            qz_pred = "SUFFICIENT" if res_qz.sufficient else "INSUFFICIENT"
            qz_score = res_qz.probability
            qz_prov = res_qz.provenance
            qz_rat = res_qz.rationale
            qwen_zero_preds.append(qz_pred)

        # 3. Qwen ICLR 2025 1-shot (CoT, zero oracle facts)
        qo_pred, qo_score, qo_prov, qo_rat = "SKIPPED", 0.0, "skipped", ""
        if qwen_oneshot_autorater is not None:
            chunk = Chunk("c0", "d0", "d0", 1, "s0", ctx, len(ctx.split()), len(ctx))
            res_qo = qwen_oneshot_autorater.evaluate(q, [chunk])
            qo_pred = "SUFFICIENT" if res_qo.sufficient else "INSUFFICIENT"
            qo_score = res_qo.probability
            qo_prov = res_qo.provenance
            qo_rat = res_qo.rationale
            qwen_oneshot_preds.append(qo_pred)

        # 4. Gemini ICLR 2025 1-shot (CoT, zero oracle facts)
        gem_pred, gem_score, gem_prov, gem_rat = "SKIPPED", 0.0, "skipped", ""
        if gemini_autorater is not None:
            chunk = Chunk("c0", "d0", "d0", 1, "s0", ctx, len(ctx.split()), len(ctx))
            res_gem = gemini_autorater.evaluate(q, [chunk])
            gem_pred = "SUFFICIENT" if res_gem.sufficient else "INSUFFICIENT"
            gem_score = res_gem.probability
            gem_prov = res_gem.provenance
            gem_rat = res_gem.rationale
            gemini_preds.append(gem_pred)

        per_instance_rows.append({
            "id": qid,
            "source_dataset": rec.get("source_dataset", ""),
            "reasoning_type": rec.get("reasoning_type", ""),
            "candidate_label": cand,
            "heuristic_prediction": h_pred,
            "heuristic_score": h_score,
            "heuristic_correct": h_pred == cand,
            "qwen_zero_prediction": qz_pred,
            "qwen_zero_score": qz_score,
            "qwen_zero_correct": qz_pred == cand if qz_pred != "SKIPPED" else "",
            "qwen_oneshot_prediction": qo_pred,
            "qwen_oneshot_score": qo_score,
            "qwen_oneshot_correct": qo_pred == cand if qo_pred != "SKIPPED" else "",
            "gemini_prediction": gem_pred,
            "gemini_score": gem_score,
            "gemini_correct": gem_pred == cand if gem_pred != "SKIPPED" else "",
            "gemini_provenance": gem_prov,
            "gemini_rationale": gem_rat,
            "qwen_oneshot_rationale": qo_rat,
        })

        if idx % 20 == 0 or idx == len(records):
            print(f"Processed {idx}/{len(records)} records...")

    # Compute Metrics
    evaluators_to_report: list[tuple[str, Sequence[str]]] = [
        ("Heuristic_Judge", heuristic_preds),
    ]
    if qwen_zero_preds:
        evaluators_to_report.append(("Qwen2.5_ZeroShot_Baseline", qwen_zero_preds))
    if qwen_oneshot_preds:
        evaluators_to_report.append(("Qwen2.5_ICLR2025_OneShot", qwen_oneshot_preds))
    if gemini_preds:
        evaluators_to_report.append(("Gemini_ICLR2025_OneShot", gemini_preds))

    metrics_map: dict[str, dict[str, Any]] = {}
    print("\n" + "=" * 90)
    print("OVERALL AUTORATER PERFORMANCE COMPARISON (80 PROXY-LABELLED BENCHMARK INSTANCES)")
    print("=" * 90)
    print(f"{'Evaluator':<28} | {'Acc':<7} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'TP':<4} | {'TN':<4} | {'FP':<4} | {'FN':<4}")
    print("-" * 90)
    for name, preds in evaluators_to_report:
        m = compute_binary_metrics(candidate_labels, preds)
        metrics_map[name] = m
        print(f"{name:<28} | {m['accuracy']*100:5.2f}% | {m['precision']:6.3f} | {m['recall']:6.3f} | {m['f1']:6.3f} | {m['tp']:<4} | {m['tn']:<4} | {m['fp']:<4} | {m['fn']:<4}")

    # Dataset-wise Breakdown
    datasets = sorted({r.get("source_dataset", "Unknown") for r in records})
    print("\n" + "=" * 90)
    print("DATASET-WISE BREAKDOWN (ACCURACY %)")
    print("=" * 90)
    header = f"{'Evaluator':<28}" + "".join(f" | {ds:<14}" for ds in datasets)
    print(header)
    print("-" * len(header))
    for name, preds in evaluators_to_report:
        row = f"{name:<28}"
        for ds in datasets:
            ds_idx = [i for i, r in enumerate(records) if r.get("source_dataset") == ds]
            ds_true = [candidate_labels[i] for i in ds_idx]
            ds_pred = [preds[i] for i in ds_idx]
            ds_m = compute_binary_metrics(ds_true, ds_pred)
            row += f" | {ds_m['accuracy']*100:5.1f}% ({ds_m['tp']+ds_m['tn']}/{len(ds_idx)})"
        print(row)

    # Reasoning-type Breakdown
    reasoning_types = sorted({r.get("reasoning_type", "Unknown") for r in records})
    print("\n" + "=" * 90)
    print("REASONING TYPE BREAKDOWN (ACCURACY %)")
    print("=" * 90)
    header_rt = f"{'Evaluator':<28}" + "".join(f" | {rt:<18}" for rt in reasoning_types)
    print(header_rt)
    print("-" * len(header_rt))
    for name, preds in evaluators_to_report:
        row = f"{name:<28}"
        for rt in reasoning_types:
            rt_idx = [i for i, r in enumerate(records) if r.get("reasoning_type") == rt]
            rt_true = [candidate_labels[i] for i in rt_idx]
            rt_pred = [preds[i] for i in rt_idx]
            rt_m = compute_binary_metrics(rt_true, rt_pred)
            row += f" | {rt_m['accuracy']*100:5.1f}% ({rt_m['tp']+rt_m['tn']}/{len(rt_idx)})"
        print(row)

    # Write summary CSV
    summary_path = results_dir / "iclr2025_autorater_benchmark.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# Research Integrity Notice: Evaluated against 80 candidate proxy labels. Human validation pending."])
        writer.writerow(["# Baseline paper reference: Joren et al. (ICLR 2025) reported 93.0% Acc, 93.5% F1 on 115 human gold labels."])
        writer.writerow(["evaluator", "benchmark_split", "instances", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
        for name, m in metrics_map.items():
            writer.writerow([name, "All_80_Proxy", m["total"], round(m["accuracy"], 4), round(m["precision"], 4), round(m["recall"], 4), round(m["f1"], 4), m["tp"], m["tn"], m["fp"], m["fn"]])

        writer.writerow([])
        writer.writerow(["# Dataset-wise Accuracy Breakdown"])
        writer.writerow(["evaluator", "dataset", "instances", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
        for name, preds in evaluators_to_report:
            for ds in datasets:
                ds_idx = [i for i, r in enumerate(records) if r.get("source_dataset") == ds]
                ds_m = compute_binary_metrics([candidate_labels[i] for i in ds_idx], [preds[i] for i in ds_idx])
                writer.writerow([name, ds, len(ds_idx), round(ds_m["accuracy"], 4), round(ds_m["precision"], 4), round(ds_m["recall"], 4), round(ds_m["f1"], 4), ds_m["tp"], ds_m["tn"], ds_m["fp"], ds_m["fn"]])

        writer.writerow([])
        writer.writerow(["# Reasoning Type Accuracy Breakdown"])
        writer.writerow(["evaluator", "reasoning_type", "instances", "accuracy", "precision", "recall", "f1", "tp", "tn", "fp", "fn"])
        for name, preds in evaluators_to_report:
            for rt in reasoning_types:
                rt_idx = [i for i, r in enumerate(records) if r.get("reasoning_type") == rt]
                rt_m = compute_binary_metrics([candidate_labels[i] for i in rt_idx], [preds[i] for i in rt_idx])
                writer.writerow([name, rt, len(rt_idx), round(rt_m["accuracy"], 4), round(rt_m["precision"], 4), round(rt_m["recall"], 4), round(rt_m["f1"], 4), rt_m["tp"], rt_m["tn"], rt_m["fp"], rt_m["fn"]])

    print(f"\n✓ Saved ICLR 2025 benchmark summary to: {summary_path.relative_to(ROOT)}")

    # Write per-instance CSV
    per_inst_path = results_dir / "iclr2025_autorater_per_instance.csv"
    with open(per_inst_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "id", "source_dataset", "reasoning_type", "candidate_label",
            "heuristic_prediction", "heuristic_score", "heuristic_correct",
            "qwen_zero_prediction", "qwen_zero_score", "qwen_zero_correct",
            "qwen_oneshot_prediction", "qwen_oneshot_score", "qwen_oneshot_correct",
            "gemini_prediction", "gemini_score", "gemini_correct",
            "gemini_provenance", "gemini_rationale", "qwen_oneshot_rationale",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_instance_rows)
    print(f"✓ Saved per-instance evaluation to: {per_inst_path.relative_to(ROOT)}")
    print(f"Total time: {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
