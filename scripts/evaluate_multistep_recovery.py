#!/usr/bin/env python3
"""Evaluation Runner for Multi-Step Cross-Document Evidence Recovery (Phase 2 Corrected).

NOTE ON BENCHMARK SCOPE:
This is a synthetic proof-of-concept benchmark evaluating multi-step cross-document
reasoning chains and intermediate bridge entity discovery. It demonstrates algorithmic
correctness and bounded depth behavior (K=0, K=1, K=2, K=3); it does NOT claim
large-scale empirical validation across open-domain datasets.

Evaluates depth regimes:
- K=0: No recovery (baseline)
- K=1: Single-pass recovery (existing baseline)
- K=2: Two-step intermediate recovery
- K=3: Three-step deep cross-document recovery

Measures:
- Recovery Success Rate (% sufficient post-recovery)
- Average Hops Executed
- Evidence Expansion (new chunks gathered)
- Termination Reason Distribution (Convergence vs Max Depth vs No New Chunks)
- Average Execution Latency

Outputs:
- results/multistep_recovery_comparison_corrected.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.structured_sufficiency import StructuredSufficiencyResult
from evidencefirst_rag.chunking import Chunk
from evidencefirst_rag.recovery import AdaptiveRecoveryController, RecoveryTrace


def build_synthetic_multihop_corpus() -> tuple[list[Chunk], list[dict[str, Any]]]:
    """Build a multi-hop evaluation corpus with cross-document evidential dependencies."""
    chunks = [
        # Chain 1: TechNova subsidiary and research lead
        Chunk(
            chunk_id="tn_overview:c1",
            document_id="tn_overview",
            document_name="technova_overview.txt",
            page=1,
            section="Subsidiaries",
            text="TechNova operates an advanced cyber-defense laboratory named CybroShield Labs.",
            token_count=10,
            char_count=80,
        ),
        Chunk(
            chunk_id="cybro_staff:c1",
            document_id="cybro_staff",
            document_name="cybro_staff_directory.txt",
            page=1,
            section="Staff",
            text="CybroShield Labs is headed by Principal Scientist Dr. Aris Thorne.",
            token_count=10,
            char_count=74,
        ),
        Chunk(
            chunk_id="thorne_patent:c1",
            document_id="thorne_patent",
            document_name="patent_filings_2024.txt",
            page=1,
            section="Patents",
            text="Dr. Aris Thorne registered the patent for Autonomous Zero-Trust Verifier in Munich.",
            token_count=12,
            char_count=87,
        ),
        # Chain 2: Quantum Computing Consortium
        Chunk(
            chunk_id="project_q:c1",
            document_id="project_q",
            document_name="quantum_consortium.txt",
            page=1,
            section="Alliances",
            text="The Quantum Security Initiative was established under Director Elena Rostova.",
            token_count=10,
            char_count=78,
        ),
        Chunk(
            chunk_id="rostova_lab:c1",
            document_id="rostova_lab",
            document_name="rostova_biography.txt",
            page=1,
            section="Background",
            text="Elena Rostova established the High-Assurance Protocol Center located in Stockholm.",
            token_count=10,
            char_count=81,
        ),
        Chunk(
            chunk_id="protocol_cert:c1",
            document_id="protocol_cert",
            document_name="standards_compliance.txt",
            page=1,
            section="Certification",
            text="The High-Assurance Protocol Center issued the Tier-4 cryptographic certificate in 2023.",
            token_count=11,
            char_count=88,
        ),
        # Chain 3: NexaCore Project Borealis 3-hop
        Chunk(
            chunk_id="nexacore_partner:c1",
            document_id="nexacore_partner",
            document_name="strategic_partners.txt",
            page=1,
            section="Partners",
            text="TechNova entered a strategic joint venture with NexaCore Dynamics.",
            token_count=10,
            char_count=66,
        ),
        Chunk(
            chunk_id="nexacore_lead:c1",
            document_id="nexacore_lead",
            document_name="nexacore_projects.txt",
            page=1,
            section="Projects",
            text="NexaCore Dynamics appointed Gregory Vance as technical program director.",
            token_count=10,
            char_count=73,
        ),
        Chunk(
            chunk_id="borealis_site:c1",
            document_id="borealis_site",
            document_name="borealis_operations.txt",
            page=1,
            section="Facilities",
            text="Gregory Vance oversees the operational station named Project Borealis.",
            token_count=10,
            char_count=70,
        ),
        Chunk(
            chunk_id="borealis_iso:c1",
            document_id="borealis_iso",
            document_name="borealis_compliance.txt",
            page=1,
            section="Audits",
            text="Project Borealis completed ISO-27001 accreditation in 2022.",
            token_count=10,
            char_count=60,
        ),
        # Standalone sufficient chunk
        Chunk(
            chunk_id="tn_hq:c1",
            document_id="tn_hq",
            document_name="technova_profile.txt",
            page=1,
            section="Profile",
            text="TechNova corporate global headquarters is located in Zurich, Switzerland.",
            token_count=10,
            char_count=74,
        ),
        # Distractor chunks
        Chunk(
            chunk_id="distract:c1",
            document_id="distract",
            document_name="general_industry_news.txt",
            page=1,
            section="News",
            text="Cloud security investments expanded across European research centers in recent years.",
            token_count=11,
            char_count=86,
        ),
        Chunk(
            chunk_id="distract:c2",
            document_id="distract",
            document_name="general_industry_news.txt",
            page=1,
            section="News",
            text="Several defense laboratories published whitepapers regarding emerging threats.",
            token_count=10,
            char_count=79,
        ),
    ]

    test_cases = [
        {
            "id": "mh-000",
            "name": "0-hop (initially sufficient)",
            "question": "Where is TechNova corporate global headquarters located?",
            "initial_chunk_id": "tn_hq:c1",
            "steps": [
                {"name": "headquarters location", "terms": ["zurich", "headquarters"], "missing_desc": "TechNova corporate headquarters location"}
            ],
        },
        {
            "id": "mh-001",
            "name": "1-hop factual bridge",
            "question": "Who is the principal scientist heading the cyber-defense laboratory operated by TechNova?",
            "initial_chunk_id": "tn_overview:c1",
            "steps": [
                {"name": "cyber-defense lab", "terms": ["cybroshield", "cyber-defense"], "missing_desc": "cyber-defense laboratory name"},
                {"name": "lab leadership", "terms": ["aris thorne", "principal scientist"], "missing_desc": "principal scientist heading CybroShield Labs"},
            ],
        },
        {
            "id": "mh-002",
            "name": "2-hop cross-doc chain (patent city)",
            "question": "In which city was the patent registered by the head of TechNova cyber-defense lab?",
            "initial_chunk_id": "tn_overview:c1",
            "steps": [
                {"name": "cyber-defense lab", "terms": ["cybroshield", "cyber-defense"], "missing_desc": "cyber-defense laboratory name"},
                {"name": "lab leadership", "terms": ["aris thorne", "head"], "missing_desc": "head of CybroShield Labs"},
                {"name": "patent city", "terms": ["munich", "patent"], "missing_desc": "city where Dr. Aris Thorne registered patent"},
            ],
        },
        {
            "id": "mh-003",
            "name": "2-hop cross-doc chain (cert year)",
            "question": "During which calendar period was the certificate issued by the organization established by Elena Rostova?",
            "initial_chunk_id": "project_q:c1",
            "steps": [
                {"name": "initiative director", "terms": ["elena rostova"], "missing_desc": "director of Quantum Security Initiative"},
                {"name": "organization established", "terms": ["high-assurance protocol center"], "missing_desc": "organization established by Elena Rostova"},
                {"name": "certificate year", "terms": ["2023", "certificate"], "missing_desc": "issuance year of certificate by High-Assurance Protocol Center"},
            ],
        },
        {
            "id": "mh-004",
            "name": "3-hop deep chain (ISO accreditation year)",
            "question": "In what year did the operational station overseen by NexaCore Dynamics program director receive accreditation?",
            "initial_chunk_id": "nexacore_partner:c1",
            "steps": [
                {"name": "joint venture partner", "terms": ["nexacore dynamics"], "missing_desc": "joint venture partner NexaCore Dynamics"},
                {"name": "program director", "terms": ["gregory vance", "program director"], "missing_desc": "technical program director of NexaCore Dynamics"},
                {"name": "station overseen", "terms": ["project borealis", "borealis"], "missing_desc": "operational station overseen by Gregory Vance"},
                {"name": "accreditation year", "terms": ["2022", "iso-27001"], "missing_desc": "accreditation year for Project Borealis"},
            ],
        },
        {
            "id": "mh-005",
            "name": "Unanswerable chain (safe termination)",
            "question": "What is the patent registration number for the Chief Financial Officer of CybroShield Labs?",
            "initial_chunk_id": "tn_overview:c1",
            "steps": [
                {"name": "cyber-defense lab", "terms": ["cybroshield"], "missing_desc": "cyber-defense laboratory name"},
                {"name": "cfo identity", "terms": ["chief financial officer", "cfo"], "missing_desc": "Chief Financial Officer of CybroShield Labs"},
                {"name": "patent registration", "terms": ["patent registration number"], "missing_desc": "patent registration number for CybroShield CFO"},
            ],
        },
    ]

    return chunks, test_cases


def run_evaluation(results_dir: Path | None = None) -> dict[int, dict[str, Any]]:
    print("=" * 80)
    print("EVIDENCEFIRST RAG: MULTI-STEP CROSS-DOCUMENT RECOVERY EVALUATION (CORRECTED)")
    print("BENCHMARK NOTE: Synthetic proof-of-concept benchmark for evidential chaining.")
    print("Evaluating depth regimes: K=0 (None), K=1 (Single-step), K=2 (Two-step), K=3 (Three-step)")
    print("=" * 80)

    corpus, test_cases = build_synthetic_multihop_corpus()
    chunks_by_id = {c.chunk_id: c for c in corpus}

    def mock_retriever(query: str, top_k: int = 6) -> list[Chunk]:
        q_tokens = set(re.findall(r"[a-z0-9\-]+", query.lower()))
        scored = []
        for c in corpus:
            c_tokens = set(re.findall(r"[a-z0-9\-]+", c.text.lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap > 0:
                scored.append((c, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def create_stepwise_judge_for_case(case: dict[str, Any]):
        """Evaluates evidential dependency chain step-by-step WITHOUT leaking future entities."""
        def judge(question: str, evidence: Sequence[Chunk]) -> StructuredSufficiencyResult:
            ev_text = " ".join(c.text.lower() for c in evidence)
            for step in case["steps"]:
                if not any(t in ev_text for t in step["terms"]):
                    desc = step["missing_desc"]
                    return StructuredSufficiencyResult(
                        sufficient=False,
                        probability=0.2,
                        missing_information=[desc],
                        rationale=f"Missing immediate evidential link: {desc}",
                    )
            return StructuredSufficiencyResult(
                sufficient=True,
                probability=1.0,
                missing_information=[],
                rationale="All multi-hop evidential links verified.",
            )
        return judge

    depths = [0, 1, 2, 3]
    summary_by_depth: dict[int, dict[str, Any]] = {}

    for depth in depths:
        controller = AdaptiveRecoveryController(max_depth=depth)
        successes = 0
        total_hops = 0
        total_new_chunks = 0
        termination_counts: dict[str, int] = {}
        latencies = []

        for case in test_cases:
            initial_chunk = chunks_by_id[case["initial_chunk_id"]]
            judge_func = create_stepwise_judge_for_case(case)

            # Initial evaluation
            init_assessment = judge_func(case["question"], [initial_chunk])
            t0 = time.perf_counter()

            if init_assessment.sufficient:
                final_suff = True
                hops = 0
                new_c = 0
                term = "ALREADY_SUFFICIENT"
            else:
                res_chunks, final_res, trace = controller.execute_recovery(
                    question=case["question"],
                    initial_chunks=[initial_chunk],
                    missing_information=init_assessment.missing_information,
                    retriever_func=mock_retriever,
                    judge_func=judge_func,
                )
                final_suff = trace.post_recovery_sufficiency
                hops = trace.total_hops_executed
                new_c = len(trace.recovered_chunks)
                term = trace.termination_reason

            latencies.append((time.perf_counter() - t0) * 1000.0)
            if final_suff:
                successes += 1
            total_hops += hops
            total_new_chunks += new_c
            termination_counts[term] = termination_counts.get(term, 0) + 1

        n = len(test_cases)
        summary_by_depth[depth] = {
            "depth_k": depth,
            "test_cases": n,
            "resolved_count": successes,
            "success_rate": round(successes / n, 4),
            "avg_hops": round(total_hops / n, 2),
            "avg_new_chunks": round(total_new_chunks / n, 2),
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "terminations": termination_counts,
        }

    print(f"{'Recovery Depth':<15} | {'Success Rate':<14} | {'Resolved':<10} | {'Avg Hops':<10} | {'Avg New Chunks':<16} | {'Avg Latency (ms)':<16}")
    print("-" * 88)
    for depth in depths:
        s = summary_by_depth[depth]
        print(
            f"K={depth:<13} | {s['success_rate']*100:6.1f}%       | {s['resolved_count']}/{s['test_cases']:<8} | "
            f"{s['avg_hops']:<10.2f} | {s['avg_new_chunks']:<16.2f} | {s['avg_latency_ms']:<16.2f}"
        )

    if results_dir is None:
        results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "multistep_recovery_comparison_corrected.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_k", "test_cases", "resolved_cases", "success_rate", "avg_hops", "avg_new_chunks", "avg_latency_ms", "terminations", "benchmark_nature"])
        for depth in depths:
            s = summary_by_depth[depth]
            term_str = "; ".join(f"{k}:{v}" for k, v in s["terminations"].items())
            writer.writerow([s["depth_k"], s["test_cases"], s["resolved_count"], s["success_rate"], s["avg_hops"], s["avg_new_chunks"], s["avg_latency_ms"], term_str, "synthetic_proof_of_concept"])

    print(f"\n✓ Saved corrected multi-step recovery evaluation to: {out_csv.relative_to(ROOT)}")
    return summary_by_depth


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation Runner for Multi-Step Cross-Document Evidence Recovery (Phase 2 Corrected).\n"
            "Evaluates bounded depth regimes K=0, K=1, K=2, K=3 on a synthetic multi-hop corpus."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory to save evaluation CSV (default: results/).",
    )
    args = parser.parse_args()
    run_evaluation(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
