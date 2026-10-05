# EvidenceFirst RAG: Dataset & Corpus Data Card

## 1. Overview & Provenance

This data card documents the corpus, evaluation tracks, and processed retrieval artifacts used by **EvidenceFirst RAG**.

- **System**: EvidenceFirst RAG
- **Authors**: Team EvidenceFirst (Avni — Retrieval & Data Lead; Darshit — Adaptive & Trust Lead; Kanika — Sufficiency & Evaluation Lead)
- **Corpus frozen**: 5 October 2026 (checksums in `data/corpus/manifest.csv`, build config in `data/processed/manifest.json`)

---

## 2. Demo Corpus (Person 1 — Avni)

| Field | Value |
|:---|:---|
| Domain | AI risk management, trustworthy AI, and cybersecurity risk guidance |
| Publisher | U.S. National Institute of Standards and Technology (NIST) |
| Documents | 24 PDFs, 910 pages, ~320k words |
| Licence / usage | U.S. Government works, not subject to copyright in the United States (17 U.S.C. §105). Free to copy and redistribute; NIST asks for attribution. A few documents reproduce third-party figures, which keep their own terms. |
| Source | `https://nvlpubs.nist.gov/` (one URL per file in the manifest) |
| Retrieved | 5 October 2026 |
| Storage | PDFs are **not committed** (33 MB). `python scripts/download_corpus.py` restores them and checks each SHA-256 against the manifest. |

Why this corpus: the documents are focused on one domain but come in different kinds (frameworks, profiles, quick-start guides, taxonomies, glossaries). They also overlap a lot in vocabulary: "risk", "profile", "tier" and "govern" appear in many documents, which makes retrieval realistically hard and makes partial-evidence and unanswerable questions easy to write. `pdf_created` is the creation date in each PDF's metadata, not the official publication date.

| File | Title | pdf_created | Pages | Chunks |
|:---|:---|:---|---:|---:|
| `nist.ai.100-1.pdf` | Artificial Intelligence Risk Management Framework (AI RMF 1.0) | 2023-01-24 | 48 | 49 |
| `NIST.AI.100-2e2025.pdf` | Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations | 2025-03-27 | 127 | 199 |
| `NIST.AI.100-3.pdf` | The Language of Trustworthy AI: An In-Depth Glossary of Terms | 2023-03-29 | 10 | 10 |
| `NIST.AI.100-4.pdf` | Reducing Risks Posed by Synthetic Content: An Overview of Technical Approaches to Digital Content Transparency | 2024-11-21 | 81 | 128 |
| `NIST.AI.100-5.pdf` | A Plan for Global Engagement on AI Standards | 2024-07-25 | 38 | 51 |
| `NIST.AI.600-1.pdf` | Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile | 2024-08-05 | 64 | 81 |
| `NIST.AI.800-1.ipd.pdf` | Managing Misuse Risk for Dual-Use Foundation Models | 2024-08-05 | 27 | 36 |
| `NIST.CSWP.29.pdf` | The NIST Cybersecurity Framework (CSF) 2.0 | 2024-03-06 | 32 | 36 |
| `NIST.IR.8286.pdf` | Integrating Cybersecurity and Enterprise Risk Management (ERM) | 2020-10-15 | 75 | 119 |
| `NIST.IR.8312.pdf` | Four Principles of Explainable Artificial Intelligence | 2021-09-29 | 43 | 77 |
| `NIST.IR.8332-draft.pdf` | Trust and Artificial Intelligence | 2021-03-19 | 30 | 45 |
| `NIST.SP.1270.pdf` | Towards a Standard for Identifying and Managing Bias in Artificial Intelligence | 2022-03-24 | 86 | 130 |
| `NIST.SP.1299.pdf` | NIST Cybersecurity Framework 2.0: Resource & Overview Guide | 2024-02-22 | 8 | 9 |
| `NIST.SP.1300.pdf` | NIST Cybersecurity Framework 2.0: Small Business Quick-Start Guide | 2024-02-22 | 9 | 9 |
| `NIST.SP.1301.pdf` | NIST Cybersecurity Framework 2.0: Quick-Start Guide for Creating and Using Organizational Profiles | 2024-02-22 | 10 | 10 |
| `NIST.SP.1302.pdf` | NIST Cybersecurity Framework 2.0: Quick-Start Guide for Using the CSF Tiers | 2024-10-17 | 3 | 3 |
| `NIST.SP.1303.pdf` | NIST Cybersecurity Framework 2.0: Enterprise Risk Management Quick-Start Guide | 2024-10-15 | 8 | 10 |
| `NIST.SP.1305.pdf` | NIST Cybersecurity Framework 2.0: Quick-Start Guide for Cybersecurity Supply Chain Risk Management (C-SCRM) | 2024-10-15 | 7 | 9 |
| `NIST.SP.1308.pdf` | NIST Cybersecurity Framework 2.0: Cybersecurity, Enterprise Risk Management, and Workforce Management Quick-Start Guide | 2026-03-19 | 11 | 13 |
| `NIST.SP.800-207.pdf` | Zero Trust Architecture | 2020-08-10 | 59 | 95 |
| `NIST.SP.800-218.pdf` | Secure Software Development Framework (SSDF) Version 1.1: Recommendations for Mitigating the Risk of Software Vulnerabilities | 2022-01-31 | 36 | 65 |
| `NIST.SP.800-218A.pdf` | Secure Software Development Practices for Generative AI and Dual-Use Foundation Models: An SSDF Community Profile | 2024-07-25 | 30 | 41 |
| `NIST.SP.800-221A.pdf` | Information and Communications Technology (ICT) Risk Outcomes: Integrating ICT Risk Management Programs with the Enterprise Risk Portfolio | 2023-11-08 | 20 | 22 |
| `NIST.SP.800-61r3.pdf` | Incident Response Recommendations and Considerations for Cybersecurity Risk Management: A CSF 2.0 Community Profile | 2025-04-01 | 48 | 52 |

The three synthetic NovaTech files in `data/demo_documents/` remain the team's small toy corpus for the Streamlit demo and unit tests. They are not part of the processed NIST index.

---

## 3. Processing Pipeline

```
PDF ─► PyMuPDF text + font sizes ─► running header/footer removal ─► page cleaning ─► 550/80 chunks ─► dense windows + BM25
```

**Ingestion** (`src/evidencefirst_rag/ingestion.py`)
- Text is extracted per page with PyMuPDF (pypdf as a fallback), so every page keeps its 1-based page number and file name.
- Running headers and footers are removed. Candidates are the first and last 6 lines of each page; a line counts as running if it recurs on at least half of the pages (and at least 3), either verbatim or with a number that advances with the page (`Page 3`, `NIST AI 100-1 … 12`). Running lines are peeled off from each edge inward, so repeated sentences inside the body are kept.
- Each page is cleaned: NFKC normalization fixes ligatures (`ﬁ` → `fi`), soft hyphens are removed, words hyphenated across lines are re-joined, and standalone page numbers, roman numerals, copyright lines, and blank lines are dropped.
- Section headings are detected from font size (≥1.15× the document's body size, or bold at body size). Markdown `#` and short ALL-CAPS titles are detected for text files.
- `document_id` is a stable slug of the file name (`NIST.AI.100-1.pdf` → `nist-ai-100-1`).
- Known limitation: one PDF (`NIST.AI.100-3.pdf`) has a broken font map that drops the `fi` ligature entirely (`artifcial`). This is left as is, because guessing missing letters would change the source text.

**Chunking** (`src/evidencefirst_rag/chunking.py`)
- Windows are exactly **550 tokens** and consecutive windows share exactly **80 tokens**. Tokens are word runs and individual punctuation marks (regex `\w+|[^\w\s]`). Unit tests check both numbers.
- Chunks never cross a page, so every chunk has exactly one page number. Text is sliced from the page, so line breaks are preserved.
- Every chunk carries `chunk_id`, `document_id`, `document_name`, `page`, `section`, `text`, `token_count`, and `char_count`.
- `section` is the nearest heading at or before the chunk start. It carries over from earlier pages of the same document.
- `chunk_id = <document_id>:chunk-<n>`, numbered per document, so ids stay stable when documents are added or removed.
- Result: 1,299 chunks; median 404 tokens, maximum 550. Short chunks are the tail ends of pages.

**Retrieval** (`src/evidencefirst_rag/retrieval.py`)

| Stage | Implementation |
|:---|:---|
| Dense | `BAAI/bge-small-en-v1.5` (384-dim, normalized, query instruction prefix) in a FAISS `IndexFlatIP` (cosine) |
| Encoder windows | A 550-token chunk is 600–800 word pieces, longer than the 512 that bge-small and the MiniLM cross-encoder read. 43% of chunks would have their tail silently truncated. Each chunk is therefore embedded as overlapping 300-token windows (50 overlap) carrying its section heading, and the chunk's score is its best window. This gives 2,117 vectors for 1,299 chunks. |
| Keyword | Okapi BM25 (k1 = 1.5, b = 0.75) over section heading + chunk text. It is rebuilt from the chunks at load time in under a second, so it is not persisted. |
| Fusion | Reciprocal Rank Fusion, k = 60, over the top 50 dense and top 50 BM25 chunks |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores the top 30 fused chunks (max over windows) and the top **6** are returned |
| Dedup | Chunks with identical text are returned once. `exclude_chunk_ids` skips chunks already in hand, so recovery retrieval brings new evidence. |
| Fallbacks | Without `pip install -e .[retrieval]`: feature-hashed embeddings, brute-force inner product, and a lexical reranker. `backend_name` on each component reports what is actually running. |

---

## 4. Sufficiency Benchmark Track (Sufficient-Context Set)

- **Total Benchmark Examples**: 80 curated and derived records across 4 public QA datasets:
  1. **PopQA** (Mallen et al., 2023) — Entity-centric open-domain queries with Wikidata entity IDs.
  2. **FreshQA** (Vu et al., 2023) — Fast-changing and false-premise world knowledge queries.
  3. **Natural Questions** (Kwiatkowski et al., 2019) — Real user search queries from Google Search.
  4. **EntityQuestions** (Sciavolino et al., 2021) — Entity-relation queries across diverse Wikidata relations.
- **Controlled Context Strategy**: For each source question, paired controlled contexts are constructed:
  - One candidate **SUFFICIENT** context variant.
  - One candidate **INSUFFICIENT** context variant (omitting critical bridging facts or premises).
- **Annotation Status**:
  - `gold_label`: Strictly initialized to `PENDING` awaiting independent human adjudication.
  - Silver Labels: Generated by independent LLM evaluation in `data/evaluation/independent_llm_annotations.jsonl`.
  - Blinded Human Subset: 40 deterministically sampled records in `data/evaluation/human_verification_subset.jsonl`.

---

## 5. Retrieval Evaluation Track (Person 1 — Avni)

- **File**: `data/eval/retrieval_eval.jsonl`. It has 64 questions over the NIST corpus: 54 answerable and 10 unanswerable (the topic is in the corpus but the asked-for fact is not).
- **Answerable question types**: 22 definition, 16 factoid, 10 list, 3 numeric, 1 temporal, and 2 multi-hop (each needing evidence from two documents).
- **Gold evidence** is stored as verbatim quotes (`document_name`, `page`, `quote`) rather than as chunk ids. At evaluation time each quote is resolved to every chunk that contains it (whitespace, quote marks and dashes are normalized). Labels therefore survive re-chunking, and a quote that matches nothing raises an error instead of silently scoring zero. When the same fact appears in several documents (for example, the CSF Tiers appear in CSWP 29 and SP 1302), the evidence is written as `any_of`.
- **Metrics**: each evidence item is a group of chunks. A group is found when any of its chunks is retrieved. Recall@k is the share of a question's groups found in the top k, averaged over questions. MRR is the reciprocal rank of the first gold chunk in the top 10.
- **How it was made**: the questions were drafted with AI assistance from the extracted text. Every quote is checked automatically against the chunks (`tests/test_retrieval_data_layer.py`). Questions are paraphrased so they do not copy the quote word for word. The set has **not** been double-checked by a second person yet; the unanswerable items especially should be reviewed by hand before they are used as gold labels.

### Retrieval ablation (`python scripts/evaluate_retrieval.py`)

Measured on 54 answerable questions over the frozen 1,299-chunk index (MacBook Air M-series CPU; latency is the median per query and includes query encoding):

| Configuration | Recall@5 | Recall@10 | MRR | Median ms |
|:---|---:|---:|---:|---:|
| Dense-only (FAISS) | 0.6574 | 0.7963 | 0.4052 | 6.5 |
| BM25-only (Okapi) | 0.7315 | 0.8519 | 0.5529 | 0.9 |
| Hybrid (RRF Fusion) | 0.7500 | 0.8704 | 0.5113 | 7.0 |
| Hybrid + Reranker (Cross-Encoder) | 0.7407 | 0.8611 | 0.6428 | 257.6 |

**Reading the table.**
- Hybrid fusion gives the best Recall@5 and Recall@10. RRF finds evidence that only one of dense or BM25 ranks highly.
- The cross-encoder mainly improves **ranking**: MRR goes from 0.51 to 0.64, so the gold chunk moves up the list. Its Recall@10 is slightly lower than plain RRF because it reorders only the top 30 fused candidates.
- BM25 is a strong baseline on this corpus because questions reuse NIST terminology.
- Per-question scores and the top 5 chunk ids are in `results/retrieval_per_query.csv`.

**Encoder-window finding.** In the first measurement, chunks were embedded whole and truncated at 512 word pieces. Dense-only reached Recall@10 = 0.72 and MRR = 0.36, and hybrid + reranker reached MRR = 0.56. The change to 300-token encoder windows was motivated by measuring truncation with the tokenizer, not by tuning on these questions.

**Retrieval misses** (hybrid + reranker, Recall@10 < 1):
- Most misses are document-level "what is/does X" questions, where cover pages and front matter that repeat the document title outrank the abstract or definition (`ret-002`, `ret-041`, `ret-048`).
- Short quick-start guides lose to related guides that share their vocabulary (`ret-028`, `ret-034`).
- One of the two multi-hop questions finds only one of its two documents (`ret-054`).
- Possible next steps: down-weight title or front-matter pages, or add document titles as a separate field.

---

## 6. Processed Artifacts

Build with `python scripts/download_corpus.py && python scripts/ingest.py`.

| Artifact | Format | Description |
| :--- | :--- | :--- |
| `data/corpus/manifest.csv` | CSV | Corpus manifest: file, title, URL, licence, page count, SHA-256 |
| `data/processed/chunks.parquet` | Parquet | 1,299 chunks with full provenance |
| `data/processed/chunks.jsonl` | JSON Lines | The same rows, loadable without pandas/pyarrow |
| `data/processed/faiss.index` | FAISS binary | `IndexFlatIP` over 2,117 window embeddings (384-dim) |
| `data/processed/faiss.index.ids.json` | JSON | Chunk id for each vector row |
| `data/processed/manifest.json` | JSON | Frozen build config: input checksums, chunk parameters, model backends |
| `data/eval/retrieval_eval.jsonl` | JSON Lines | Retrieval evaluation questions with verbatim evidence quotes |
| `results/retrieval_ablation.csv` | CSV | Recall@5, Recall@10, MRR, and latency per configuration |
| `results/retrieval_per_query.csv` | CSV | Per-question scores and top-5 chunk ids per configuration |

Use from code:

```python
from evidencefirst_rag.retrieval import HybridRetriever
retriever = HybridRetriever.from_processed("data/processed")   # reuses faiss.index if backends match
top6 = retriever.retrieve("What are the four CSF Tiers?")       # Chunk objects with page + section
```

---

## 7. Ethical Considerations & Limitations

- **No personal data.** The NIST documents are public guidance. Author names appear only in the documents' own front matter.
- **Labels are not human gold.** The retrieval questions are AI-assisted drafts whose evidence was checked automatically. See section 4 for the sufficiency benchmark's own labelling status.
- **Coverage.** The corpus is English-only, U.S.-government guidance from 2020–2026. Retrieval results should not be read as general open-domain performance.
- **Drafts.** Two documents are drafts or initial public drafts (`NIST.IR.8332-draft`, `NIST.AI.800-1.ipd`) and may differ from final versions.
