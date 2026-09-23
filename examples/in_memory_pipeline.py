"""EvidenceFirst RAG demo: sufficient context, recovery, and abstention."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag import (
    AgenticRAGPipeline,
    CorpusDescriptor,
    CoverageSufficiencyJudge,
    ExtractiveSynthesizer,
    GroundedAnswer,
    InMemoryDocument,
    InMemoryKeywordRetriever,
    KeywordPlanner,
    TemplateQueryRewriter,
)


def _print_evidencefirst_trace(
    question: str,
    answer: GroundedAnswer,
    documents: Sequence[InMemoryDocument],
) -> None:
    doc_text_by_id = {doc.id: doc.text for doc in documents}
    iterations = answer.audit.get("iterations", [])
    if not iterations:
        return

    first_it = iterations[0]
    plan = first_it.get("plan") or {}
    required_facts = plan.get("required_facts") or ()
    routes = {
        r.get("fact_id"): r.get("candidate_corpus_ids", ())
        for r in plan.get("routes", ())
    }

    print("-" * 70)
    print("EVIDENCEFIRST TRACE")
    print("-" * 70)

    # 1. Question
    print(f"1. Question:\n   {question}\n")

    # 2. Required facts
    print("2. Required facts:")
    for fact in required_facts:
        f_id = fact.get("id", "")
        f_desc = fact.get("description", "")
        target = ", ".join(routes.get(f_id, ()))
        print(f"   - [{f_id}] \"{f_desc}\" -> routed to corpus: '{target}'")
    print()

    # 3. Retrieved evidence / source corpus
    print("3. Retrieved evidence / source corpus:")
    first_subqueries = first_it.get("subqueries") or []
    first_hits = first_it.get("hits") or []
    first_assessment = first_it.get("assessment") or {}

    for sq in first_subqueries:
        sq_id = sq.get("id", "")
        sq_query = sq.get("query", "")
        sq_corpus = ", ".join(sq.get("target_corpus_ids", ()))
        sq_hits = [
            h for h in first_hits
            if h.get("metadata", {}).get("subquery_id") == sq_id
        ]
        print(f"   - Subquery '{sq_id}' (\"{sq_query}\") -> corpus '{sq_corpus}':")
        if sq_hits:
            for h in sq_hits:
                doc_id = h.get("document_id", "")
                text = doc_text_by_id.get(doc_id, "")
                score = h.get("score", 0.0)
                print(f"     * Hit '{h.get('id')}' (doc: {doc_id}, score: {score:.2f}): \"{text}\"")
        else:
            print(f"     * No hits returned (evidence unavailable in corpus '{sq_corpus}')")

    covered_facts = {
        cf.get("fact_id"): cf.get("snippet_ids", ())
        for cf in first_assessment.get("covered_facts", ())
    }
    summary_parts = []
    for fact in required_facts:
        f_id = fact.get("id", "")
        f_desc = fact.get("description", "")
        f_corpus = ", ".join(routes.get(f_id, ()))
        if f_id in covered_facts:
            summary_parts.append(f"\"{f_desc}\" retrieved from '{f_corpus}' corpus")
        else:
            summary_parts.append(f"\"{f_desc}\" missing from '{f_corpus}' corpus")
    print(f"   -> Status: {'; '.join(summary_parts)}.")
    print()

    # 4. Sufficiency decision
    status_str = str(first_assessment.get("status", "")).upper()
    reason = first_assessment.get("reason", "")
    print(f"4. Sufficiency decision:\n   {status_str} ({reason})\n")

    # 5. Sufficiency score
    score = first_assessment.get("sufficiency_score", 0.0)
    print(f"5. Sufficiency score:\n   {score:.2f}\n")

    # 6. Recovery attempt(s), when applicable
    print("6. Recovery attempt(s):")
    if len(iterations) > 1:
        for it in iterations[1:]:
            it_num = it.get("iteration", 2)
            subqueries = it.get("subqueries") or []
            print(f"   [Iteration {it_num} - Follow-up Retrieval]")
            for sq in subqueries:
                sq_query = sq.get("query", "")
                sq_corpus = ", ".join(sq.get("target_corpus_ids", ()))
                print(f"   - Feedback subquery: \"{sq_query}\" targeting corpus '{sq_corpus}'")
                feedback_hits = [
                    h for h in it.get("hits", [])
                    if h.get("metadata", {}).get("subquery_id") == sq.get("id")
                ]
                if feedback_hits:
                    for h in feedback_hits:
                        doc_id = h.get("document_id", "")
                        print(f"     * Recovered hit '{h.get('id')}' (doc: {doc_id}): \"{doc_text_by_id.get(doc_id, '')}\"")
                else:
                    print(f"     * 0 hits returned for \"{sq_query}\" in '{sq_corpus}'")
            it_assess = it.get("assessment") or {}
            it_status = str(it_assess.get("status", "")).upper()
            it_missing = it_assess.get("missing_facts", ())
            if it_missing:
                print(f"   -> Result: Follow-up retrieval complete; status remains {it_status} (unresolved: {', '.join(it_missing)}).")
            else:
                print(f"   -> Result: Follow-up retrieval resolved missing context; status: {it_status}.")
    else:
        print("   None needed (context is sufficient in iteration 1).")
    print()

    # 7. Final decision
    final_status = answer.status.value.upper() if hasattr(answer.status, "value") else str(answer.status).upper()
    if final_status in ("ANSWERED", "SUFFICIENT"):
        print(f"7. Final decision:\n   {final_status} (Context is sufficient; answer generated from verified citations)\n")
    elif answer.missing_facts:
        missing_quoted = ", ".join(repr(f) for f in answer.missing_facts)
        print(f"7. Final decision:\n   {final_status} (Selective abstention applied: missing evidence for {missing_quoted}; no unsupported facts invented)\n")
    else:
        print(f"7. Final decision:\n   {final_status} (Selective abstention applied)\n")

    # 8. Missing facts, when applicable
    print("8. Missing facts:")
    if answer.missing_facts:
        for fact in answer.missing_facts:
            print(f"   - {fact}")
    else:
        print("   None")
    print()

    # 9. Citations
    print("9. Citations:")
    if answer.citations:
        for citation in answer.citations:
            print(f"   - {citation.claim} -> {', '.join(citation.snippet_ids)}")
    else:
        print("   None")
    print()


def run_demo(
    question: str,
    documents: Sequence[InMemoryDocument] | None = None,
    scenario_label: str = "",
) -> GroundedAnswer:
    print("\n" + "=" * 70)
    if scenario_label:
        print(f"EVIDENCEFIRST RAG — {scenario_label}")
    else:
        print("EVIDENCEFIRST RAG")
    print("=" * 70)

    if documents is None:
        documents = (
            InMemoryDocument(
                "company-1",
                "company",
                "TechNova was founded by Alice Sharma.",
            ),
            InMemoryDocument(
                "company-2",
                "company",
                "TechNova is headquartered in Bangalore.",
            ),
            InMemoryDocument(
                "history-1",
                "history",
                "TechNova was founded in 2018.",
            ),
            InMemoryDocument(
                "distractor-1",
                "news",
                "TechNova released a new mobile application in 2024.",
            ),
        )

    pipeline = AgenticRAGPipeline(
        planner=KeywordPlanner(),
        query_rewriter=TemplateQueryRewriter(),
        retriever=InMemoryKeywordRetriever(documents),
        sufficiency_judge=CoverageSufficiencyJudge(),
        synthesizer=ExtractiveSynthesizer(),
        max_iterations=2,
    )

    corpora = (
        CorpusDescriptor(
            "company",
            "Company organization who founder founders leadership executive headquarters facts",
        ),
        CorpusDescriptor(
            "history",
            "Historical timeline founding year dates when events chronological company history",
        ),
        CorpusDescriptor(
            "news",
            "Recent news announcements product releases",
        ),
    )

    answer = pipeline.answer(question, corpora)

    _print_evidencefirst_trace(question, answer, documents)

    print("-" * 70)
    print("RESULT")
    print("-" * 70)
    print(f"Answer: {answer.answer}")
    print(f"Status: {answer.status}")
    print(f"Sufficiency score: {answer.sufficiency_score}")
    print(f"Iterations: {answer.iterations}")

    if answer.missing_facts:
        print("\nMissing facts:")
        for fact in answer.missing_facts:
            print(f"  - {fact}")

    print("\nCitations:")
    if answer.citations:
        for citation in answer.citations:
            print(
                f"  - {citation.claim} "
                f"-> {', '.join(citation.snippet_ids)}"
            )
    else:
        print("  None")

    print("=" * 70)
    return answer


def main() -> None:
    question = "Who founded TechNova and what year was it founded?"

    # CASE A — SUFFICIENT:
    # Both the founder fact (company corpus) and the founding year fact (history corpus) are available.
    documents_case_a = (
        InMemoryDocument(
            "company-1",
            "company",
            "TechNova was founded by Alice Sharma.",
        ),
        InMemoryDocument(
            "company-2",
            "company",
            "TechNova is headquartered in Bangalore.",
        ),
        InMemoryDocument(
            "history-1",
            "history",
            "TechNova was founded in 2018.",
        ),
        InMemoryDocument(
            "distractor-1",
            "news",
            "TechNova released a new mobile application in 2024.",
        ),
    )
    run_demo(
        question,
        documents=documents_case_a,
        scenario_label="CASE A — SUFFICIENT",
    )

    # CASE B — INSUFFICIENT:
    # The founding year evidence is unavailable. The pipeline attempts recovery across iterations,
    # detects the missing fact, and abstains (PARTIAL) without hallucinating a year.
    documents_case_b = (
        InMemoryDocument(
            "company-1",
            "company",
            "TechNova was founded by Alice Sharma.",
        ),
        InMemoryDocument(
            "company-2",
            "company",
            "TechNova is headquartered in Bangalore.",
        ),
        InMemoryDocument(
            "distractor-1",
            "news",
            "TechNova released a new mobile application in 2024.",
        ),
    )
    run_demo(
        question,
        documents=documents_case_b,
        scenario_label="CASE B — INSUFFICIENT",
    )


if __name__ == "__main__":
    main()
