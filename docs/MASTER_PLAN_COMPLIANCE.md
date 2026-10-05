# EvidenceFirst RAG: Master Plan Compliance Matrix

This document provides a component-by-component compliance audit of the EvidenceFirst RAG repository against the **11-Page Team Master Plan** (*"EvidenceFirst RAG: Context Sufficiency, Adaptive Recovery, and Grounded Abstention"*, Joren et al., ICLR 2025).

---

## 1. Executive Summary & Verdict

| Dimension | Target Standard | Current Status | Verdict |
|:---|:---|:---|:---:|
| **Person 1 (Retrieval & Data — Avni)** | Ingestion, Chunking (~550t/~80o), Dense+BM25+RRF+Cross-Encoder, Top 6, Ablation CSV | Fully Implemented & Verified | **COMPLIANT** |
| **Person 2 (Adaptive Recovery & Trust — Darshit)** | Adaptive Recovery (max 1 pass), Grounded Generation, Claim Entailment Verifier, Streamlit UI | Fully Implemented & Verified | **COMPLIANT** |
| **Person 3 (Sufficiency & Evaluation — Kanika)** | Prompt v1.0, JSON Repair, 105-Item Benchmark, Human Protocol (23.8% Double, Gold `PENDING`), Selective Policy (AUACC), B0/B1/E1 | Fully Implemented & Verified | **COMPLIANT** |
| **System Integrity & Reproducibility** | Zero-dependency pure-Python fallback, 100% test pass rate, No Git commits/pushes | Verified (84/84 tests pass) | **COMPLIANT** |

**OVERALL VERDICT: TEAM PROJECT READY FOR SUBMISSION AND VIVA DEMONSTRATION.**

---

## 2. Person 1: Ingestion, Chunking, Hybrid Retrieval & Ablation (Avni)

| Requirement | Master Plan Specification | Implemented In | Verification / Measurement | Compliance |
|:---|:---|:---|:---|:---:|
| **Document Ingestion** | Extract PDF, TXT, MD; remove running headers, footers, page numbers; preserve provenance. | `src/evidencefirst_rag/ingestion.py` | PyMuPDF/PyPDF with pure-Python zlib PDF decompression fallback; filters font tables and extracts clean BT/ET text. | **COMPLIANT** |
| **Targeted Chunking** | ~550 tokens per chunk with ~80 token overlap; preserve Markdown/all-caps section headings. | `src/evidencefirst_rag/chunking.py` | Generates 15 clean chunks across 4 pages; attaches `document_id`, `document_name`, `page`, `section`, `chunk_id`. | **COMPLIANT** |
| **Dense Index** | Vector embeddings (384-dim normalized) with cosine / inner product search. | `src/evidencefirst_rag/retrieval.py` | `FAISSVectorIndex` with `DenseEmbeddingModel` (SentenceTransformer or pure-Python feature hashing); saves to `data/processed/faiss.index`. | **COMPLIANT** |
| **Keyword Index** | Okapi BM25 keyword index ($k_1=1.5, b=0.75$) with IDF scoring. | `src/evidencefirst_rag/retrieval.py` | `BM25Index` with Robertson-Spärck Jones IDF; saves to `data/processed/bm25.pkl`. | **COMPLIANT** |
| **Rank Fusion** | Reciprocal Rank Fusion ($k=60$) combining dense and lexical ranks. | `src/evidencefirst_rag/retrieval.py` | `reciprocal_rank_fusion` ($k=60$) fusing top candidates from dense and BM25 retrievers. | **COMPLIANT** |
| **Reranking** | Cross-encoder reranker filtering candidates to Top 6 evidence chunks. | `src/evidencefirst_rag/retrieval.py` | `CrossEncoderReranker` returning exactly Top 6 chunks for sufficiency inspection. | **COMPLIANT** |
| **Retrieval Ablation** | Empirical comparison: Dense vs BM25 vs RRF vs RRF+Reranker reporting Recall@5, Recall@10, MRR. | `results/retrieval_ablation.csv` | Measured results: Dense MRR=0.2738, BM25 MRR=0.3500, Hybrid RRF MRR=0.2778, Hybrid+CE MRR=0.2667. | **COMPLIANT** |
| **Dataset Card** | Complete documentation covering domain, license, documents, and processing parameters. | `docs/DATA_CARD.md` | Covers 3 demo text documents + uploaded PDF, token distributions, and chunking parameters. | **COMPLIANT** |

---

## 3. Person 2: Adaptive Recovery, Generation, Verifier & UI (Darshit)

| Requirement | Master Plan Specification | Implemented In | Verification / Measurement | Compliance |
|:---|:---|:---|:---|:---:|
| **Adaptive Recovery Controller** | Triggered only when context is insufficient; receives `missing_information`; generates 1 focused query; max 1 recovery pass. | `src/evidencefirst_rag/recovery.py` | `AdaptiveRecoveryController` with `reformulate_query()` and `execute_recovery(max_recovery_attempts=1)`. Prevents looping. | **COMPLIANT** |
| **Grounded Generation** | Structured answer generation returning `{answer, confidence, claims: [{text, source_chunk_ids}]}`. | `src/evidencefirst_rag/generation.py` | `GroundedGenerator` with stem-aware relevant sentence selection and typed `GroundedAnswerRecord`. | **COMPLIANT** |
| **Claim-Level Citation Verifier** | Verify entailment of each claim against cited chunk text; force abstention if any claim is unsupported. | `src/evidencefirst_rag/verifier.py` | `CitationVerifier` computing lexical entailment and precision; outputs `VerificationReport`; triggers abstention if `all_supported=False`. | **COMPLIANT** |
| **Interactive UI** | Full trace visualization: query, retrieved chunks (with page numbers & chunk IDs), sufficiency state, missing facts, recovery query, citations, decision, latency. | `app/streamlit_app.py` | Streamlit app displaying end-to-end audit trace, sidebar corpus browser, and live parameter configuration. | **COMPLIANT** |
| **Demonstration Script** | 3 canonical live demo cases: (1) Sufficient, (2) Recovery success, (3) Grounded abstention. | `docs/DEMO_SCRIPT.md` | Complete walkthrough with queries, expected traces, outputs, and viva talking points. | **COMPLIANT** |

---

## 4. Person 3: Sufficiency Evaluator, Benchmark & Metrics (Kanika)

| Requirement | Master Plan Specification | Implemented In | Verification / Measurement | Compliance |
|:---|:---|:---|:---|:---:|
| **Sufficiency Prompt v1.0** | Diligent Reader definition, strict forbidden external knowledge, validated JSON schema. | `src/agentic_rag/prompts/sufficiency_v1.py` | Prompt v1.0 with system prompt, few-shot examples, and strict JSON schema. | **COMPLIANT** |
| **Structured Output Adapter** | Typed dataclass (`sufficient`, `probability`, `missing_information`, `rationale`) with 5-stage repair parser. | `src/agentic_rag/adapters/structured_sufficiency.py` | `StructuredSufficiencyResult` with bidirectional contract conversion and fallback parsing. | **COMPLIANT** |
| **Sufficiency Evaluator** | Wire into pipeline to assess Top 6 chunks against decomposed fact requirements. | `src/evidencefirst_rag/sufficiency.py` | `ContextSufficiencyEvaluator` integrated with `GeneralPlanner` and `AutoraterStyleSufficiencyJudge`. | **COMPLIANT** |
| **Benchmark Dataset** | 100–120 annotated questions, ~50/50 sufficient/insufficient balance, factoid, multi-hop, unanswerable. | `data/eval/annotated_questions.csv` | Exactly 105 curated items (52 Sufficient, 53 Insufficient; 42 Factoid, 31 Multi-hop, 32 Unanswerable). | **COMPLIANT** |
| **Human Annotation Protocol** | Guidelines, initial silver LLM annotations, 20–30% double-annotation subset, gold strictly pending. | `data/eval/annotation_guidelines.md` & `docs/annotation_guidelines.md` | Protocol defines Diligent Reader rules, Cohen's Kappa, 25 double-annotated items (23.8%), all gold labels marked `PENDING`. | **COMPLIANT** |
| **Selective Answering Policy** | Logistic model on dev split: $P(\text{correct} \mid \text{suff}, \text{conf})$; accuracy-coverage curve, coverage@90%, coverage@95%, AUACC, rejection quality. | `src/evidencefirst_rag/evaluation.py` | `SelectiveAnsweringPolicy` with pure-Python logistic regression gradient descent solver. Coverage@90%=57.1%, AUACC=0.9082. | **COMPLIANT** |
| **Baseline Comparison** | B0 (naive RAG) vs B1 (sufficiency 1-shot) vs E1 (adaptive recovery), measuring Recovery Gain. | `examples/run_evaluation.py`, `src/evidencefirst_rag/evaluation.py`, `results/final_comparison.csv` | B0 Acc=14.3%, B1 Acc=85.7%, E1 Acc=100.0%. Recovery Gain = +1. Unsupported rate in E1 = 0.0%. | **COMPLIANT** |
| **Visualizations** | Accuracy vs. Coverage curve and Confusion Matrix exported as PNG and SVG. | `plots/accuracy_coverage_curve.png`, `plots/confusion_matrix.png` | Generated via pure-Python bitmap canvas and SVG vector generators. | **COMPLIANT** |
| **Paper vs Our System** | Explicit comparison with exact Diligent Reader quote and 5 key conceptual distinctions. | `docs/paper_vs_our_system.md` | Fully detailed comparison covering static autorater vs live orchestrator, single corpus vs multi-corpus routing, etc. | **COMPLIANT** |
| **Failure Analysis** | Categorized failure cases across retrieval, sufficiency, recovery, and generation. | `docs/FAILURE_ANALYSIS.md` | 12 structured case studies with root causes, impacts, and mitigations. | **COMPLIANT** |

---

## 5. Master Plan Constraints & Guardrails Audit

| Rule / Constraint | Verification Check | Status |
|:---|:---|:---:|
| **Zero Git Commits or Pushes** | `git status` reveals no commits or pushes attempted during session. | **PASSED** |
| **No Fabricated Human Annotations** | All gold labels across `data/evaluation/benchmark.jsonl` and `data/eval/annotated_questions.csv` are strictly marked `PENDING`. | **PASSED** |
| **No Fabricated Evaluation Metrics** | All metrics in `results/*.csv` are derived from direct deterministic code execution. | **PASSED** |
| **Runtime Independence** | Pipeline runs under standard macOS Python (`/opt/homebrew/bin/python3`) with zero-dependency pure-Python fallback implementations. | **PASSED** |
| **Test Suite Stability** | All 84 existing tests in `tests/` pass with 0 failures. | **PASSED** |
