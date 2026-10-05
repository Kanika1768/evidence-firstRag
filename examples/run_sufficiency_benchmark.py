"""EvidenceFirst Independent Sufficient-Context Benchmark Runner.

Evaluates context sufficiency detection at the retrieval-generation boundary
using the EvidenceFirst Sufficient-Context detector across four public QA pools:
- PopQA (Mallen et al., 2023)
- FreshQA (Vu et al., 2023)
- Natural Questions (Kwiatkowski et al., 2019)
- EntityQuestions (Sciavolino et al., 2021)

Adheres strictly to the three-tier separation:
1. Source-data curation
2. Gold-label annotation workflow (diligent reader standard)
3. Model evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.contracts import (  # noqa: E402
    Claim,
    ContextAssessment,
    ContextStatus,
    Corpus,
    DraftAnswer,
    FactPriority,
    RequiredFact,
    RetrievalPlan,
    Route,
    Snippet,
)
from agentic_rag.general_planner import GeneralPlanner  # noqa: E402
from agentic_rag.sufficiency import AutoraterStyleSufficiencyJudge  # noqa: E402

DEFAULT_BENCHMARK_PATH = ROOT / "data" / "evaluation" / "benchmark.jsonl"


@dataclass(frozen=True)
class ExampleEvaluationResult:
    """Detailed evaluation result for a single benchmark record."""

    record_id: str
    source_dataset: str
    source_record_id: str | None
    source_locator: str
    original_question: str
    source_qa_answer: str
    context_source_type: str
    context_snippet: str
    gold_label: str
    candidate_label: str
    predicted_label: str
    sufficiency_score: float
    reasoning_type: str
    covered_facts: tuple[str, ...]
    missing_facts: tuple[str, ...]
    rationale: str

    @property
    def is_gold_annotated(self) -> bool:
        return self.gold_label in {"SUFFICIENT", "INSUFFICIENT"}

    @property
    def matches_gold(self) -> bool | None:
        if not self.is_gold_annotated:
            return None
        return self.predicted_label == self.gold_label

    @property
    def matches_candidate(self) -> bool:
        return self.predicted_label == self.candidate_label


def load_benchmark(benchmark_path: Path) -> list[dict[str, Any]]:
    """Load benchmark records from JSONL."""
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found at {benchmark_path}")

    records: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON at line {line_no} of {benchmark_path}: {exc}"
                ) from exc
    return records


def build_evaluation_plan(
    record: Mapping[str, Any],
    planner: GeneralPlanner,
    corpus: Corpus,
    mode: str = "planner",
) -> RetrievalPlan:
    """Build a RetrievalPlan for evaluating the question + context pair.

    Modes:
    - 'planner': Uses GeneralPlanner to deterministically decompose the question.
    - 'fact_grounded': Uses the curated required_facts list from the benchmark record.
    """
    question = record["original_question"]

    if mode == "planner":
        return planner.plan(question, [corpus])

    # fact_grounded mode: build plan from curated required facts
    facts: list[RequiredFact] = []
    stopwords = {"who", "what", "when", "where", "why", "how", "the", "a", "an", "and", "or", "of", "to", "in", "is", "was"}
    for idx, fact_desc in enumerate(record.get("required_facts", [question]), start=1):
        terms = [
            t for t in re.findall(r"[a-z0-9]+", fact_desc.lower())
            if len(t) > 2 and t not in stopwords
        ]
        facts.append(
            RequiredFact(
                id=f"f{idx}",
                description=fact_desc,
                priority=FactPriority.MUST,
                metadata={"required_terms": tuple(terms)},
            )
        )

    routes = tuple(
        Route(
            fact_id=f.id,
            candidate_corpus_ids=(corpus.id,),
            reason="Routed to benchmark context corpus.",
        )
        for f in facts
    )

    return RetrievalPlan(
        question=question,
        required_facts=tuple(facts),
        routes=routes,
        stop_conditions=(),
    )


def evaluate_example(
    record: Mapping[str, Any],
    judge: AutoraterStyleSufficiencyJudge,
    planner: GeneralPlanner,
    corpus: Corpus,
    mode: str = "planner",
) -> ExampleEvaluationResult:
    """Run sufficiency assessment on a single (question, context) benchmark instance."""
    question = record["original_question"]
    context = record["context"]

    plan = build_evaluation_plan(record, planner, corpus, mode=mode)

    # Provide the context as a retrieved snippet attributed to the corpus
    snippet = Snippet(
        id=f"snip-{record['id']}",
        corpus_id=corpus.id,
        document_id=f"doc-{record['id']}",
        text=context,
        score=1.0,
    )

    # Draft claims based on available context text
    draft = DraftAnswer(
        claims=(Claim(text=context[:120].strip(), snippet_ids=(snippet.id,)),),
        cited_snippet_ids=(snippet.id,),
    )

    assessment: ContextAssessment = judge.assess(
        question=question,
        plan=plan,
        snippets=[snippet],
        draft=draft,
    )

    pred_status = (
        "SUFFICIENT"
        if assessment.status == ContextStatus.SUFFICIENT
        else "INSUFFICIENT"
    )

    covered_ids = tuple(cf.fact_id for cf in assessment.covered_facts)

    return ExampleEvaluationResult(
        record_id=record["id"],
        source_dataset=record["source_dataset"],
        source_record_id=record.get("source_record_id"),
        source_locator=record.get("source_locator", "unspecified"),
        original_question=question,
        source_qa_answer=record.get("source_qa_answer", ""),
        context_source_type=record.get("context_source_type", "constructed_controlled_context"),
        context_snippet=context if len(context) <= 110 else f"{context[:107]}...",
        gold_label=record.get("gold_label", "PENDING"),
        candidate_label=record.get("candidate_label", "UNKNOWN"),
        predicted_label=pred_status,
        sufficiency_score=assessment.sufficiency_score,
        reasoning_type=record.get("reasoning_type", "unknown"),
        covered_facts=covered_ids,
        missing_facts=tuple(assessment.missing_facts),
        rationale=record.get("rationale", ""),
    )


def compute_binary_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = "SUFFICIENT",
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have identical length")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp == positive_label)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp != positive_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != positive_label and yp == positive_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == positive_label and yp != positive_label)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total": float(total),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_benchmark(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    mode: str = "planner",
    verbose: bool = False,
) -> dict[str, Any]:
    """Execute the benchmark evaluation and print formatted reports."""
    records = load_benchmark(benchmark_path)
    planner = GeneralPlanner()
    corpus = Corpus(
        id="eval_corpus",
        description="Benchmark reference context passage.",
    )
    judge = AutoraterStyleSufficiencyJudge()

    results: list[ExampleEvaluationResult] = [
        evaluate_example(r, judge, planner, corpus, mode=mode)
        for r in records
    ]

    print("=" * 76)
    print("EVIDENCEFIRST INDEPENDENT SUFFICIENT-CONTEXT BENCHMARK REPORT")
    print("Inspired by ICLR 2025: Joren et al., 'Sufficient Context: A New Lens on RAG'")
    print("=" * 76)

    # 1. Dataset & Annotation Status
    total_count = len(results)
    gold_annotated = [res for res in results if res.is_gold_annotated]
    pending_count = total_count - len(gold_annotated)

    dataset_counts: dict[str, int] = {}
    for res in results:
        dataset_counts[res.source_dataset] = dataset_counts.get(res.source_dataset, 0) + 1

    print("\n1. BENCHMARK DATASET & ANNOTATION STATUS")
    print("-" * 76)
    print(f"Total Benchmark Instances : {total_count}")
    print(f"Verified Human Gold Labels: {len(gold_annotated)}")
    print(f"Pending Annotations       : {pending_count} (awaiting independent human review)")
    print("Source Dataset Distribution:")
    for ds_name, count in sorted(dataset_counts.items()):
        print(f"  - {ds_name:18}: {count} examples")

    # 2. Gold Metric Evaluation (if any verified gold labels exist)
    print("\n2. MODEL EVALUATION AGAINST VERIFIED GOLD LABELS")
    print("-" * 76)
    if gold_annotated:
        gold_true = [res.gold_label for res in gold_annotated]
        gold_pred = [res.predicted_label for res in gold_annotated]
        gold_metrics = compute_binary_metrics(gold_true, gold_pred)
        print(f"Annotated Subset Size : {len(gold_annotated)}")
        print(f"Accuracy              : {gold_metrics['accuracy']:.3f} ({int(gold_metrics['tp'] + gold_metrics['tn'])}/{len(gold_annotated)})")
        print(f"Precision (SUFFICIENT): {gold_metrics['precision']:.3f}")
        print(f"Recall    (SUFFICIENT): {gold_metrics['recall']:.3f}")
        print(f"F1-Score  (SUFFICIENT): {gold_metrics['f1']:.3f}")
        print(f"Confusion Matrix      : TP={int(gold_metrics['tp'])}, TN={int(gold_metrics['tn'])}, FP={int(gold_metrics['fp'])}, FN={int(gold_metrics['fn'])}")
    else:
        print("STATUS: All 80 records have gold_label='PENDING'.")
        print("Independent human annotation workflow is outlined in data/evaluation/README.md.")
        print("To compute verified gold metrics, populate gold labels via annotation_template.jsonl.")

    # 3. Candidate Alignment Analysis (Development Heuristic)
    print("\n3. CURATOR CANDIDATE ALIGNMENT (DEVELOPMENT HEURISTIC)")
    print("-" * 76)
    print("[NOTE] Candidate labels reflect curated development expectations and are NOT")
    print("       claimed as independent human gold ground truth.")
    cand_true = [res.candidate_label for res in results]
    cand_pred = [res.predicted_label for res in results]
    cand_metrics = compute_binary_metrics(cand_true, cand_pred)

    print(f"Candidate Label Alignment: {cand_metrics['accuracy'] * 100:.1f}% ({int(cand_metrics['tp'] + cand_metrics['tn'])}/{total_count})")
    print(f"Candidate Precision       : {cand_metrics['precision']:.3f}")
    print(f"Candidate Recall          : {cand_metrics['recall']:.3f}")
    print(f"Candidate F1-Score        : {cand_metrics['f1']:.3f}")
    print(f"Candidate Confusion Matrix: TP={int(cand_metrics['tp'])}, TN={int(cand_metrics['tn'])}, FP={int(cand_metrics['fp'])}, FN={int(cand_metrics['fn'])}")

    # 4. Breakdown by Source Dataset
    print("\n4. CANDIDATE ALIGNMENT BREAKDOWN BY SOURCE DATASET")
    print("-" * 76)
    print(f"{'Source Dataset':<20} | {'Total':<6} | {'Agreement':<10} | {'Pred SUFF':<10} | {'Pred INSUFF':<11}")
    print("-" * 76)
    dataset_breakdown: dict[str, dict[str, Any]] = {}
    for ds_name in sorted(dataset_counts.keys()):
        ds_results = [r for r in results if r.source_dataset == ds_name]
        ds_true = [r.candidate_label for r in ds_results]
        ds_pred = [r.predicted_label for r in ds_results]
        metrics = compute_binary_metrics(ds_true, ds_pred)
        pred_suff = sum(1 for p in ds_pred if p == "SUFFICIENT")
        pred_insuff = sum(1 for p in ds_pred if p == "INSUFFICIENT")
        dataset_breakdown[ds_name] = {
            "total": len(ds_results),
            "agreement": metrics["accuracy"],
            "pred_sufficient": pred_suff,
            "pred_insufficient": pred_insuff,
        }
        print(f"{ds_name:<20} | {len(ds_results):<6} | {metrics['accuracy'] * 100:6.1f}%    | {pred_suff:<10} | {pred_insuff:<11}")

    # 5. Breakdown by Reasoning Type
    print("\n5. CANDIDATE ALIGNMENT BREAKDOWN BY REASONING TYPE")
    print("-" * 76)
    reasoning_types = sorted({r.reasoning_type for r in results})
    print(f"{'Reasoning Type':<30} | {'Total':<6} | {'Agreement':<10} | {'Candidate Dist':<15}")
    print("-" * 76)
    for rtype in reasoning_types:
        rt_results = [r for r in results if r.reasoning_type == rtype]
        rt_true = [r.candidate_label for r in rt_results]
        rt_pred = [r.predicted_label for r in rt_results]
        metrics = compute_binary_metrics(rt_true, rt_pred)
        cand_suff = sum(1 for t in rt_true if t == "SUFFICIENT")
        cand_insuff = sum(1 for t in rt_true if t == "INSUFFICIENT")
        print(f"{rtype:<30} | {len(rt_results):<6} | {metrics['accuracy'] * 100:6.1f}%    | S:{cand_suff} / I:{cand_insuff}")

    # 6. Diagnostic Traces
    print("\n6. SAMPLE EVIDENCEFIRST SUFFICIENCY DIAGNOSTIC TRACES")
    print("-" * 76)
    sample_indices = [0, 1, 20, 21, 40, 41, 60, 61] if not verbose else list(range(total_count))
    for idx in sample_indices:
        if idx >= len(results):
            continue
        res = results[idx]
        src_id_display = (
            f"Src ID: {res.source_record_id}"
            if res.source_record_id is not None
            else f"Locator: {res.source_locator} (Upstream ID: null)"
        )
        print(f"[{res.record_id}] Source: {res.source_dataset} ({src_id_display}) | Context: {res.context_source_type}")
        print(f"  Q       : {res.original_question}")
        print(f"  Source A: {res.source_qa_answer}")
        print(f"  Context : \"{res.context_snippet}\"")
        print(f"  Decision: Predicted={res.predicted_label} (Score={res.sufficiency_score:.2f}) | Candidate={res.candidate_label} | Gold={res.gold_label}")
        if res.missing_facts:
            print(f"  Missing : {list(res.missing_facts)}")
        print(f"  Rationale: {res.rationale}")
        print()

    # 7. Comparison to ICLR 2025 Paper
    print("7. RELATION TO ICLR 2025 EXPERIMENTAL FINDINGS")
    print("-" * 76)
    print("In 'Sufficient Context: A New Lens on RAG' (Joren et al., 2025):")
    print("- Evaluated 115 human-labeled QA instances across diverse domains.")
    print("- Found that standard RAG systems frequently attempt answers even when")
    print("  retrieved context is insufficient, yielding ungrounded hallucinations.")
    print("- The EvidenceFirst architecture guards the generation stage: when the judge")
    print("  detects missing required facts, it triggers recovery reformulation or selective")
    print("  abstention, ensuring grounded output.")
    # 8. Export metrics to CSV
    metrics_csv_path = export_sufficiency_metrics_csv(cand_metrics)
    print(f"\n✓ Exported sufficiency metrics table to: {metrics_csv_path}")

    return {
        "total_records": total_count,
        "verified_gold_records": len(gold_annotated),
        "pending_records": pending_count,
        "candidate_metrics": cand_metrics,
        "dataset_breakdown": dataset_breakdown,
        "results": results,
    }


def export_sufficiency_metrics_csv(
    metrics: Mapping[str, float],
    output_path: Path = ROOT / "results" / "sufficiency_metrics.csv",
) -> Path:
    """Export benchmark evaluation metrics to CSV with AUROC=N/A documentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Total Instances", str(int(metrics.get("total", 80))), "Complete benchmark dataset"),
        ("True Positives (TP)", str(int(metrics.get("tp", 0))), "Correctly predicted SUFFICIENT"),
        ("True Negatives (TN)", str(int(metrics.get("tn", 0))), "Correctly predicted INSUFFICIENT"),
        ("False Positives (FP)", str(int(metrics.get("fp", 0))), "Predicted SUFFICIENT but candidate was INSUFFICIENT"),
        ("False Negatives (FN)", str(int(metrics.get("fn", 0))), "Predicted INSUFFICIENT but candidate was SUFFICIENT"),
        ("Accuracy", f"{metrics.get('accuracy', 0.0):.4f}", "Alignment with candidate/curated labels"),
        ("Precision", f"{metrics.get('precision', 0.0):.4f}", "High precision ensures minimal hallucinations"),
        ("Recall", f"{metrics.get('recall', 0.0):.4f}", "Conservative Diligent Reader threshold"),
        ("F1-Score", f"{metrics.get('f1', 0.0):.4f}", "Harmonic mean of precision and recall"),
        ("AUROC", "N/A", "Judge outputs binary decisions {1.0, 0.0}; continuous scores required for AUROC calculation"),
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_name", "value", "note"])
        for row in rows:
            writer.writerow(row)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EvidenceFirst Independent Sufficient-Context Benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="Path to benchmark JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=["planner", "fact_grounded"],
        default="planner",
        help="Planning strategy: 'planner' uses GeneralPlanner; 'fact_grounded' uses curated facts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic traces for all benchmark examples.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to dump structured evaluation summary.",
    )
    args = parser.parse_args()

    summary = run_benchmark(
        benchmark_path=args.benchmark,
        mode=args.mode,
        verbose=args.verbose,
    )

    if args.output_json:
        export_payload = {
            "total_records": summary["total_records"],
            "verified_gold_records": summary["verified_gold_records"],
            "pending_records": summary["pending_records"],
            "candidate_metrics": summary["candidate_metrics"],
            "dataset_breakdown": summary["dataset_breakdown"],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"\nStructured results exported to {args.output_json}")


if __name__ == "__main__":
    main()
