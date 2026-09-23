from pathlib import Path

from agentic_rag.contracts import Corpus
from agentic_rag.document_loader import load_documents
from agentic_rag.chunking import chunk_documents
from agentic_rag.document_index import build_local_index
from agentic_rag.generic_pipeline import build_generic_components
from agentic_rag.orchestrator import (
    AgenticRAGOrchestrator,
    OrchestratorConfig,
)


DATA_DIR = Path("data/demo_documents")


def build_corpora(documents):
    corpora = []

    for document in documents:
        corpus_id = Path(document.document_name).stem

        description = (
            f"{document.document_name} "
            f"{document.text[:500]}"
        )

        corpora.append(
            Corpus(
                id=corpus_id,
                description=description,
            )
        )

    return tuple(corpora)




class DemoRecoveryRewriter:
    """Demo-only rewriter that hides finance on the first retrieval pass."""

    def __init__(self, base_rewriter):
        self.base_rewriter = base_rewriter

    def rewrite(self, question, plan, prior_assessment, iteration):
        queries = tuple(
            self.base_rewriter.rewrite(
                question,
                plan,
                prior_assessment,
                iteration,
            )
        )

        if question.strip() == "What is NovaTech's annual revenue?":
            if prior_assessment is None:
                # Keep the initial query, but search only the non-finance corpora.
                queries = tuple(
                    type(query)(
                        id=query.id,
                        fact_id=query.fact_id,
                        query=query.query,
                        target_corpus_ids=tuple(
                            corpus_id
                            for corpus_id in query.target_corpus_ids
                            if corpus_id != "finance"
                        ) or ("company", "history"),
                        reason=query.reason,
                        parent_query=query.parent_query,
                        iteration=query.iteration,
                    )
                    for query in queries
                )
            else:
                # Recovery pass: explicitly target the hidden finance corpus.
                queries = tuple(
                    type(query)(
                        id=query.id,
                        fact_id=query.fact_id,
                        query=query.query,
                        target_corpus_ids=("finance",),
                        reason=query.reason,
                        parent_query=query.parent_query,
                        iteration=query.iteration,
                    )
                    for query in queries
                )

        return queries

def print_result(question, result):
    answer = result.answer

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    print("\n" + "-" * 80)
    print("FINAL ANSWER")
    print("-" * 80)
    print(answer.answer)

    print("\nSTATUS:", answer.status)
    print("SUFFICIENCY SCORE:", answer.sufficiency_score)
    print("ITERATIONS:", len(result.iterations))

    if answer.missing_facts:
        print("\nMISSING FACTS:")
        for fact in answer.missing_facts:
            print(" -", fact)

    print("\nCITATIONS:")
    if answer.citations:
        for citation in answer.citations:
            print(" -", citation.claim)
            print("   snippets:", citation.snippet_ids)
    else:
        print(" - None")

    print("\nRETRIEVED EVIDENCE:")
    if result.snippets:
        for snippet in result.snippets:
            print(
                f" - [{snippet.corpus_id}/{snippet.document_id}] "
                f"score={snippet.score:.2f}"
            )
            print(f"   {snippet.text}")
    else:
        print(" - None")

    print("\nTRACE:")

    for trace in result.iterations:
        assessment = trace.assessment

        print(
            f"Iteration {trace.iteration + 1}: "
            f"status={assessment.status}, "
            f"score={assessment.sufficiency_score}"
        )

        print("  Queries:")
        for query in trace.subqueries:
            print(f"   - {query.query}")

        if assessment.missing_facts:
            print("  Missing:", assessment.missing_facts)

        if assessment.feedback_queries:
            print("  Recovery queries:")
            for feedback in assessment.feedback_queries:
                print(f"   - {feedback.query}")

def main():
    print("Loading documents...")

    documents = load_documents(DATA_DIR)

    print(f"Documents loaded: {len(documents)}")

    for document in documents:
        print(
            f" - {document.document_name} "
            f"(page {document.page})"
        )

    chunks = chunk_documents(
        documents,
        chunk_size=800,
        overlap=120,
    )

    print(f"\nChunks created: {len(chunks)}")

    retriever = build_local_index(
        documents,
        chunk_size=800,
        overlap=120,
        per_query_limit=5,
    )

    corpora = build_corpora(documents)

    print("\nCorpora:")

    for corpus in corpora:
        print(f" - {corpus.id}")

    components = build_generic_components()

    rewriter = DemoRecoveryRewriter(components["rewriter"])

    orchestrator = AgenticRAGOrchestrator(
        planner=components["planner"],
        rewriter=rewriter,
        retriever=retriever,
        drafter=components["drafter"],
        judge=components["judge"],
        synthesizer=components["synthesizer"],
        config=OrchestratorConfig(
            max_iterations=2,
        ),
    )

    questions = [
        ("Who founded NovaTech and when was it founded?", corpora),
        ("What does NovaTech develop?", corpora),
        ("What is NovaTech's annual revenue?", corpora),
    ]

    for question, question_corpora in questions:
        result = orchestrator.run(
            question,
            question_corpora,
        )

        print_result(question, result)


if __name__ == "__main__":
    main()
