import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.retriever import LEXICAL_SCORING_RULE, LexicalDocument, LexicalRetriever  # noqa: E402
from agentic_rag.contracts import Snippet, Subquery  # noqa: E402


class LexicalRetrieverTests(unittest.TestCase):
    def test_documents_scoring_rule(self):
        self.assertIn("overlapping query terms", LEXICAL_SCORING_RULE)
        self.assertIn("exact phrase", LEXICAL_SCORING_RULE)

    def test_retrieves_only_routed_corpora_and_preserves_provenance(self):
        retriever = LexicalRetriever(
            (
                LexicalDocument(
                    corpus_id="projects",
                    document_id="project-1",
                    text="Project Zen owner is Nina.",
                    metadata={"source": "project-db", "version": 3},
                ),
                LexicalDocument(
                    corpus_id="noise",
                    document_id="noise-1",
                    text="Project Zen owner is not listed in cafeteria notices.",
                    metadata={"source": "cafeteria"},
                ),
            )
        )
        query = Subquery(
            id="q1-0",
            fact_id="project_owner",
            query="Project Zen owner",
            target_corpus_ids=("projects",),
            reason="Project records contain owners.",
            parent_query="Who owns Alice's project?",
            iteration=1,
        )

        snippets = retriever.retrieve(query)

        self.assertEqual(1, len(snippets))
        snippet = snippets[0]
        self.assertIsInstance(snippet, Snippet)
        self.assertEqual("projects", snippet.corpus_id)
        self.assertEqual("project-1", snippet.document_id)
        self.assertEqual("project-db", snippet.metadata["source"])
        self.assertEqual(3, snippet.metadata["version"])
        self.assertGreater(snippet.score, 0)
        self.assertEqual("q1-0", snippet.query_id)
        self.assertEqual("project_owner", snippet.fact_id)
        self.assertIsNotNone(snippet.span)
        self.assertEqual("Project Zen owner", snippet.text[snippet.span[0] : snippet.span[1]])
        self.assertEqual(("projects",), retriever.calls[0].target_corpus_ids)

    def test_returns_empty_when_query_terms_are_not_in_routed_corpus(self):
        retriever = LexicalRetriever(
            (
                LexicalDocument("directory", "people-1", "Alice works on Project Zen."),
                LexicalDocument("projects", "project-1", "Project Zen owner is Nina."),
            )
        )
        query = Subquery(
            id="q0-0",
            fact_id="approval",
            query="Atlas approval",
            target_corpus_ids=("projects",),
            reason="Approval should be in projects.",
        )

        self.assertEqual((), retriever.retrieve(query))

    def test_orders_results_by_score_then_stable_provenance_keys(self):
        retriever = LexicalRetriever(
            (
                LexicalDocument("projects", "doc-b", "Project Zen owner."),
                LexicalDocument("projects", "doc-a", "Project Zen owner."),
                LexicalDocument("projects", "doc-c", "Project Zen owner Nina has final approval."),
            )
        )
        query = Subquery(
            id="q2-0",
            fact_id="project_owner",
            query="Project Zen owner Nina",
            target_corpus_ids=("projects",),
            reason="Project records contain owners.",
        )

        snippets = retriever.retrieve(query)

        self.assertEqual(("doc-c", "doc-a", "doc-b"), tuple(snippet.document_id for snippet in snippets))
        self.assertGreater(snippets[0].score, snippets[1].score)
        self.assertEqual(snippets[1].score, snippets[2].score)

    def test_extracts_overlap_span_when_exact_phrase_is_absent(self):
        retriever = LexicalRetriever((LexicalDocument("projects", "doc-1", "Nina leads the Zen migration."),))
        query = Subquery(
            id="q3-0",
            fact_id="project_owner",
            query="Zen Nina",
            target_corpus_ids=("projects",),
            reason="Project records contain owners.",
        )

        snippet = retriever.retrieve(query)[0]

        self.assertEqual("Nina leads the Zen", snippet.text[snippet.span[0] : snippet.span[1]])

    def test_deduplicates_documents_by_corpus_and_document_id(self):
        retriever = LexicalRetriever(
            (
                LexicalDocument("projects", "project-1", "Project Zen owner is Nina.", {"version": "old"}),
                LexicalDocument(
                    "projects",
                    "project-1",
                    "Project Zen owner is Nina and the sponsor is Omar.",
                    {"version": "new"},
                ),
            )
        )
        query = Subquery(
            id="q4-0",
            fact_id="project_owner",
            query="Project Zen owner sponsor",
            target_corpus_ids=("projects",),
            reason="Project records contain owners.",
        )

        snippets = retriever.retrieve(query)

        self.assertEqual(1, len(snippets))
        self.assertEqual("project-1", snippets[0].document_id)
        self.assertEqual("new", snippets[0].metadata["version"])


if __name__ == "__main__":
    unittest.main()
