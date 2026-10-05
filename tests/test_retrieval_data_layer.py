"""Unit tests for the Person 1 (Avni) retrieval & data layer.

Covers ingestion provenance and header/footer cleaning, exact chunk size and
overlap, page/section metadata, stable chunk ids, hybrid retrieval behaviour,
index persistence, and the retrieval metrics used in the ablation.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evidencefirst_rag.chunking import Chunk, chunk_document_pages, chunk_page, count_tokens
from evidencefirst_rag.ingestion import (
    DocumentPage,
    clean_page_text,
    load_document,
    make_document_id,
    remove_repeated_edge_lines,
)
from evidencefirst_rag.retrieval import (
    ENCODER_WINDOW_TOKENS,
    RETRIEVAL_MODES,
    FAISSVectorIndex,
    HybridRetriever,
    RetrievalEvalCase,
    encoder_windows,
    evaluate_retrieval_ablation,
    load_processed_chunks,
    load_retrieval_eval_cases,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
    resolve_evidence_chunk_ids,
)

ROOT = Path(__file__).resolve().parents[1]

try:
    import pymupdf
except ImportError:
    pymupdf = None


def _page(text: str, page: int = 1, doc: str = "doc-a", headings: tuple[str, ...] = ()) -> DocumentPage:
    return DocumentPage(doc, f"{doc}.pdf", f"data/{doc}.pdf", page, text, headings)


def _chunk(cid: str, text: str, doc: str = "doc", page: int = 1) -> Chunk:
    return Chunk(cid, doc, f"{doc}.pdf", page, "General", text, count_tokens(text), len(text))


class TestIngestion(unittest.TestCase):
    def test_repeated_headers_and_footers_removed_but_body_kept(self) -> None:
        topics = ["governance", "mapping", "measurement", "management", "oversight"]
        pages = [
            ["NIST AI 100-1", "AI RMF 1.0", "January 2023", f"Opening paragraph about {t}.", "Shared middle text.", f"Closing on {t}.", f"Page {i} of 5"]
            for i, t in enumerate(topics, start=1)
        ]
        cleaned = remove_repeated_edge_lines(pages)
        for i, (lines, topic) in enumerate(zip(cleaned, topics), start=1):
            self.assertEqual(lines, [f"Opening paragraph about {topic}.", "Shared middle text.", f"Closing on {topic}."])

    def test_numbers_that_do_not_follow_page_numbering_are_kept(self) -> None:
        pages = [[f"Table {n} summary", "Body.", "More body.", "Even more.", "Last."] for n in (4, 9, 2, 7)]
        cleaned = remove_repeated_edge_lines(pages)
        self.assertEqual([lines[0] for lines in cleaned], ["Table 4 summary", "Table 9 summary", "Table 2 summary", "Table 7 summary"])

    def test_repeat_detection_skipped_for_short_documents(self) -> None:
        pages = [["Header", "Body one."], ["Header", "Body two."]]
        self.assertEqual(remove_repeated_edge_lines(pages), pages)

    def test_clean_page_text_normalizes_ligatures_soft_hyphens_and_line_hyphenation(self) -> None:
        cleaned = clean_page_text("Arti\ufb01cial intel-\nligence and termi\u00ad\nnology\n\n12\nCopyright 2024 NIST")
        self.assertEqual(cleaned, "Artificial intelligence and terminology")

    def test_document_id_is_stable_slug(self) -> None:
        self.assertEqual(make_document_id(Path("NIST.AI.100-1.pdf")), "nist-ai-100-1")
        self.assertEqual(make_document_id(Path("company.txt")), "company")

    def test_text_file_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Policy Notes.md"
            path.write_text("# Scope\nThe policy covers all staff.\n", encoding="utf-8")
            pages = load_document(path)
        self.assertEqual(len(pages), 1)
        self.assertEqual((pages[0].document_id, pages[0].document_name, pages[0].page), ("policy-notes", "Policy Notes.md", 1))

    @unittest.skipIf(pymupdf is None, "PyMuPDF not installed")
    def test_pdf_pages_keep_page_numbers_and_headings_without_running_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Sample.Report.pdf"
            doc = pymupdf.open()
            for i in range(1, 5):
                page = doc.new_page()
                page.insert_text((72, 40), "Sample Report Running Header", fontsize=9)
                page.insert_text((72, 100), f"Overview of Topic {'ABCD'[i - 1]}", fontsize=18)
                for line_no in range(8):
                    page.insert_text((72, 140 + 16 * line_no), f"Body text line {line_no} that belongs to page {i} only.", fontsize=11)
                page.insert_text((72, 780), f"Page {i}", fontsize=9)
            doc.save(path)
            pages = load_document(path)

        self.assertEqual([p.page for p in pages], [1, 2, 3, 4])
        self.assertTrue(all(p.document_id == "sample-report" for p in pages))
        for p in pages:
            self.assertNotIn("Running Header", p.text)
            self.assertIn(f"Overview of Topic {'ABCD'[p.page - 1]}", p.headings)
            self.assertIn(f"belongs to page {p.page} only", p.text)


class TestChunking(unittest.TestCase):
    def setUp(self) -> None:
        self.words = [f"w{i}" for i in range(1200)]
        self.page = _page(" ".join(self.words), page=7)

    def test_windows_are_550_tokens_with_exact_80_token_overlap(self) -> None:
        chunks, next_idx = chunk_page(self.page, target_tokens=550, overlap_tokens=80)
        self.assertEqual(next_idx, len(chunks))
        token_lists = [c.text.split() for c in chunks]
        for tokens, chunk in zip(token_lists[:-1], chunks[:-1]):
            self.assertEqual(len(tokens), 550)
            self.assertEqual(chunk.token_count, 550)
        for prev, nxt in zip(token_lists, token_lists[1:]):
            self.assertEqual(prev[-80:], nxt[:80])
        self.assertEqual(token_lists[0][0], "w0")
        self.assertEqual(token_lists[-1][-1], "w1199")
        self.assertLessEqual(len(token_lists[-1]), 550)

    def test_every_chunk_carries_page_metadata_and_unique_id(self) -> None:
        chunks, _ = chunk_page(self.page)
        self.assertTrue(all(c.page == 7 and c.document_id == "doc-a" and c.document_name == "doc-a.pdf" for c in chunks))
        self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
        self.assertTrue(all(c.token_count == count_tokens(c.text) for c in chunks))

    def test_chunks_never_cross_pages(self) -> None:
        pages = [_page("alpha " * 300, page=1), _page("beta " * 300, page=2)]
        chunks = chunk_document_pages(pages)
        self.assertEqual([c.page for c in chunks], [1, 2])
        self.assertNotIn("beta", chunks[0].text)

    def test_overlap_must_be_smaller_than_target(self) -> None:
        with self.assertRaises(ValueError):
            chunk_page(self.page, target_tokens=80, overlap_tokens=80)

    def test_chunk_ids_restart_per_document_and_stay_stable(self) -> None:
        doc_b = [_page("gamma " * 700, page=1, doc="doc-b")]
        alone = chunk_document_pages(doc_b)
        with_other = chunk_document_pages([_page("alpha " * 900, page=1, doc="doc-a")] + doc_b)
        self.assertEqual([c.chunk_id for c in alone], [c.chunk_id for c in with_other if c.document_id == "doc-b"])
        self.assertEqual(alone[0].chunk_id, "doc-b:chunk-0")

    def test_section_comes_from_preceding_heading_and_carries_across_pages(self) -> None:
        pages = [
            _page("Intro text.\nRisk Measurement\n" + "measure " * 30, page=1, headings=("Risk Measurement",)),
            _page("continued discussion " * 20, page=2),
            _page("# Risk Tolerance\nTolerance text.", page=3),
        ]
        chunks = chunk_document_pages(pages)
        self.assertEqual(chunks[0].section, "Risk Measurement")
        self.assertEqual(chunks[1].section, "Risk Measurement")
        self.assertEqual(chunks[2].section, "Risk Tolerance")

    def test_page_text_formatting_is_preserved(self) -> None:
        chunks, _ = chunk_page(_page("Heading Line\nFirst sentence.\nSecond sentence."))
        self.assertEqual(chunks[0].text, "Heading Line\nFirst sentence.\nSecond sentence.")


class TestRetrievalComponents(unittest.TestCase):
    def test_encoder_windows_cover_long_chunks(self) -> None:
        text = " ".join(f"t{i}" for i in range(550))
        windows = encoder_windows(_chunk("c", text))
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(count_tokens(w) <= ENCODER_WINDOW_TOKENS for w in windows))
        self.assertTrue(windows[0].startswith("t0 ") and windows[-1].endswith("t549"))
        self.assertEqual(encoder_windows(_chunk("s", "short text")), ["short text"])

    def test_rrf_fuses_any_number_of_rankings(self) -> None:
        fused = reciprocal_rank_fusion([("a", 1.0), ("b", 0.5)], [("b", 9.0), ("c", 1.0)], [("b", 1.0)], k=60)
        self.assertEqual(fused[0][0], "b")
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61 + 1 / 61)

    def test_retriever_excludes_ids_dedupes_text_and_validates_mode(self) -> None:
        chunks = [
            _chunk("c1", "Zero trust removes implicit trust based on network location."),
            _chunk("c2", "Zero trust removes implicit trust based on network location."),
            _chunk("c3", "The policy engine decides whether to grant access in zero trust."),
            _chunk("c4", "Backups should be tested regularly by small businesses."),
        ]
        retriever = HybridRetriever(chunks)
        for mode in RETRIEVAL_MODES:
            ids = [c.chunk_id for c in retriever.retrieve("zero trust implicit trust network location", top_k=4, mode=mode)]
            self.assertFalse({"c1", "c2"} <= set(ids), mode)
        excluded = retriever.retrieve("zero trust implicit trust", top_k=4, exclude_chunk_ids={"c1", "c2"})
        self.assertNotIn("c1", [c.chunk_id for c in excluded])
        self.assertNotIn("c2", [c.chunk_id for c in excluded])
        with self.assertRaises(ValueError):
            retriever.retrieve("anything", mode="semantic")

    def test_vector_index_save_and_load_roundtrip(self) -> None:
        index = FAISSVectorIndex(dim=4)
        index.add_many(["a", "b", "a"], [[1, 0, 0, 0], [0, 1, 0, 0], [0.6, 0.8, 0, 0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faiss.index"
            index.save(path)
            loaded = FAISSVectorIndex.load(path)
        self.assertEqual(loaded.chunk_ids, ["a", "b", "a"])
        self.assertEqual([cid for cid, _ in loaded.search([0, 1, 0, 0], top_k=3)], ["b", "a", "a"])


class _StubRetriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def retrieve(self, query: str, top_k: int = 6, mode: str = "hybrid_rerank") -> list[Chunk]:
        return [_chunk(cid, cid) for cid in self.rankings[query][:top_k]]


class TestRetrievalMetrics(unittest.TestCase):
    def test_recall_counts_groups_satisfied_by_any_member(self) -> None:
        groups = [{"a1", "a2"}, {"b"}]
        self.assertEqual(recall_at_k(groups, ["x", "a2", "y"], 5), 0.5)
        self.assertEqual(recall_at_k(groups, ["x", "a2", "y", "z", "q", "b"], 5), 0.5)
        self.assertEqual(recall_at_k(groups, ["x", "a2", "y", "z", "q", "b"], 10), 1.0)

    def test_reciprocal_rank_uses_first_gold_hit(self) -> None:
        self.assertEqual(reciprocal_rank([{"g"}], ["x", "y", "g"]), 1 / 3)
        self.assertEqual(reciprocal_rank([{"g"}], ["x"]), 0.0)

    def test_ablation_aggregates_exact_values_and_writes_csv(self) -> None:
        cases = [RetrievalEvalCase("q1", ["g1"]), RetrievalEvalCase("q2", ["g2"])]
        stub = _StubRetriever({"q1": ["g1", "x"], "q2": ["x", "y", "z", "w", "v", "u", "g2"]})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ablation.csv"
            results = evaluate_retrieval_ablation(cases, stub, output_csv=out)
            rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        for mode in RETRIEVAL_MODES:
            self.assertEqual(results[mode]["Recall@5"], 0.5)
            self.assertEqual(results[mode]["Recall@10"], 1.0)
            self.assertEqual(results[mode]["MRR"], round((1 + 1 / 7) / 2, 4))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["n_queries"], "2")

    def test_evidence_quotes_resolve_with_normalized_quotes_and_whitespace(self) -> None:
        chunks = [_chunk("d:chunk-0", "In the AI RMF, risk refers to the composite\nmeasure of an event’s probability.")]
        self.assertEqual(resolve_evidence_chunk_ids(chunks, "doc.pdf", "composite measure of an event's probability"), ["d:chunk-0"])
        self.assertEqual(resolve_evidence_chunk_ids(chunks, "doc.pdf", "composite measure", page=2), [])

    def test_eval_loader_handles_any_of_skips_unanswerable_and_rejects_missing_quotes(self) -> None:
        chunks = [_chunk("a:chunk-0", "Alpha fact here.", doc="a"), _chunk("b:chunk-0", "Alpha fact here too.", doc="b")]
        rows = [
            {"qid": "q1", "question": "alpha?", "evidence": [{"any_of": [{"document_name": "a.pdf", "quote": "alpha fact"}, {"document_name": "b.pdf", "quote": "alpha fact"}]}]},
            {"qid": "q2", "question": "unknown?", "evidence": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            cases = load_retrieval_eval_cases(path, chunks)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].groups(), [{"a:chunk-0", "b:chunk-0"}])
            path.write_text(json.dumps({"qid": "q3", "question": "?", "evidence": [{"document_name": "a.pdf", "quote": "missing"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_retrieval_eval_cases(path, chunks)


class TestCommittedDataArtifacts(unittest.TestCase):
    def test_corpus_manifest_lists_20_to_30_checksummed_public_documents(self) -> None:
        rows = list(csv.DictReader((ROOT / "data" / "corpus" / "manifest.csv").read_text(encoding="utf-8").splitlines()))
        self.assertTrue(20 <= len(rows) <= 30)
        self.assertTrue(all(len(r["sha256"]) == 64 and r["url"].startswith("https://") for r in rows))

    def test_retrieval_eval_set_resolves_against_processed_chunks(self) -> None:
        eval_path = ROOT / "data" / "eval" / "retrieval_eval.jsonl"
        rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len({r["qid"] for r in rows}), len(rows))
        self.assertGreaterEqual(sum(r["answerable"] for r in rows), 50)
        processed = ROOT / "data" / "processed"
        if not (processed / "chunks.jsonl").exists():
            self.skipTest("data/processed not built")
        chunks = load_processed_chunks(processed)
        self.assertEqual(len(load_retrieval_eval_cases(eval_path, chunks)), sum(r["answerable"] for r in rows))


if __name__ == "__main__":
    unittest.main()
