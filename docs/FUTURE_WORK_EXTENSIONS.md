# EvidenceFirst RAG: IEEE Future-Work Extensions & Scientific Evaluation Report

This report documents the design, implementation, provenance architecture, and empirical findings for the five research extensions described in our IEEE conference paper.

---

## 1. Executive Summary & Verification Matrix

All five extensions have been implemented, covered by 183 automated tests (182 passed, 0 failures, 0 errors, 1 skipped live Ollama test), and experimentally evaluated on target corpora and benchmark suites, with experimental limitations, backend dependencies, and pending human validation explicitly documented.

| Extension Area | Implementation Files | Test Suite | Evaluation Dataset | Key Empirical Finding | Human Status |
|:---|:---|:---|:---|:---|:---:|
| **1. Human Validation Infrastructure** | `src/evidencefirst_rag/annotation_tools.py`, `scripts/manage_human_validation.py` | `tests/test_human_annotation_infrastructure.py` | 80 benchmark instances (`PopQA`, `FreshQA`, `NQ`, `EntityQuestions`) | Exported blinded CSV/JSONL packages; schema-validated annotation parser; Cohen's $\kappa$ agreement workflow. | **PENDING** (0/80 completed) |
| **2. LLM AutoRater & Provenance** | `src/agentic_rag/adapters/local_llm_client.py`, `src/agentic_rag/adapters/structured_sufficiency.py`, `scripts/evaluate_autorater.py` | `tests/test_autorater_evaluation.py` | 80 benchmark instances | Local Qwen2.5-0.5B-Instruct achieves 45.0% accuracy (80/80 results carry `qwen_local_inference` provenance, 0% fallbacks; verified cached outputs); exhibits extreme positive bias (40/40 FP on insufficient slices). Evaluator prompt is oracle-assisted with curated required facts. Heuristic judge achieves 63.8% accuracy. | Gold PENDING |
| **3. Improved Retrieval on Difficult Queries** | `src/evidencefirst_rag/retrieval.py`, `scripts/evaluate_retrieval_improvements.py` | `tests/test_retrieval_improvements.py` | 54 answerable queries on 1,287 NIST chunks (24 PDFs) | Adaptive Score Fusion improves MRR by +15.04% over RRF (0.4454 $\to$ 0.5124); Adaptive Hybrid + Reranker achieves peak Recall@5 of 0.7407. (Dense baseline evaluated as FAISS over feature-hash vectors due to local neural model unavailability). | Verified* |
| **4. Multi-Step Cross-Document Recovery** | `src/evidencefirst_rag/recovery.py`, `scripts/evaluate_multistep_recovery.py` | `tests/test_multistep_recovery.py` | 13-chunk synthetic proof-of-concept multi-hop corpus | Elimination of downstream entity leakage; sequential bridge discovery resolves 16.7% at $K=0$, 33.3% at $K=1$, 66.7% at $K=2$, and 83.3% at $K=3$ on a bounded 6-query synthetic proof-of-concept suite. | Verified* |
| **5. NIST End-to-End Comparative Evaluation** | `src/evidencefirst_rag/pipeline.py`, `src/evidencefirst_rag/sufficiency.py`, `scripts/evaluate_nist_end_to_end.py` | `tests/test_nist_end_to_end.py` | Full 64-query NIST benchmark (54 answerable + 10 unanswerable) | B0/B1/E1 evaluated under identical `hybrid_rerank` mode. E1 achieves 83.3% answerable coverage, 100.0% unanswerable abstention, and recorded zero unsupported answers against current benchmark labels (0.0% vs 15.6% in B0/B1). | Verified* |

*\*Note on "Verified"*: In this matrix, "Verified" indicates that the software implementation, automated test suites, and empirical benchmark evaluations have been completed, executed, and recorded. It does NOT denote independent human validation, which remains strictly **PENDING** across all 80 benchmark instances.

---

## 2. Phase 1: Human Validation Infrastructure

### 2.1 Design & Protocol
To eliminate reliance on proxy candidate labels without fabricating human scores, we built a formal human annotation workflow adhering to the **Diligent Reader Protocol** (Joren et al., ICLR 2025):
- **Criterion**: A retrieved context is sufficient if and only if a diligent reader can construct a definitive answer using **only** the provided text, without external world knowledge or parametric memory.
- **Blinded Export**: `scripts/manage_human_validation.py export` generates blinded packages in `data/evaluation/reviewer_packages/`. All candidate labels, curator rationales, and model predictions are completely stripped so independent reviewers cannot be biased.
- **Independent Reviews**: Each reviewer independently assigns `human_decision` (`SUFFICIENT` or `INSUFFICIENT`) and `human_rationale`.
- **Inter-Annotator Agreement**: Computes observed agreement $P_o$, expected chance agreement $P_e$, and Cohen's Kappa $\kappa = \frac{P_o - P_e}{1 - P_e}$.
- **Adjudication**: Where reviewers agree, labels are promoted to `gold_label`. Discrepancies are flagged for adjudication.

### 2.2 Current Validation Status
```bash
python scripts/manage_human_validation.py status
```
```text
======================================================================
EVIDENCEFIRST RAG: HUMAN VALIDATION STATUS REPORT
======================================================================
Benchmark File:     data/evaluation/benchmark.jsonl
Total Instances:    80
Human Gold Labels:  0 (Verified)
Pending Reviews:    80 (PENDING HUMAN REVIEW)
Completion Rate:    0.0%

Label Breakdown:
  - PENDING: 80
======================================================================
NOTE: Gold labels strictly require independent human review.
Candidate/proxy development labels are stored separately and
must not be reported as human-validated ground truth.
======================================================================
```
*Scientific Integrity Statement*: Human validation is explicitly marked as **PENDING**. Candidate labels are evaluated only as development proxies.

---

## 3. Phase 2: LLM AutoRater Evaluation & Provenance Architecture

### 3.1 Provenance Architecture & Fallback Prevention
In earlier iterations, heuristic fallback completions risked being mixed with model completions. We implemented a strict **zero-contamination provenance architecture**:
1. **Isolated Verified Cache**: Cached Qwen completions are written exclusively to `data/evaluation/.cache_autorater_qwen/` (separate from legacy unverified caches).
2. **Provenance Metadata**: Every cache entry stores the SHA-256 prompt hash, model identifier (`Qwen/Qwen2.5-0.5B-Instruct`), timestamp, and explicit tag (`qwen_local_inference`). Provenance reflects genuine local model completions and can represent previously generated, verified cached model output rather than live inference during each individual evaluation run.
3. **Strict No-Fallback Caching**: Fallback responses from deterministic rules are **never** written to the model cache directory.
4. **Explicit Mode Tagging**: When fallback triggers, `LLMSufficiencyAutoRater` records `evaluator_mode = "Fallback: Heuristic Judge (<reason>)"` and `provenance = "heuristic_fallback_on_<reason>"`. Results can never silently disguise a fallback as model inference.

### 3.2 Empirical AutoRater Results (N=80)
Evaluated across all 80 benchmark instances where 80/80 results carry `qwen_local_inference` provenance (utilizing verified cached outputs from local Qwen2.5-0.5B-Instruct):

| Evaluator | Accuracy | Precision | Recall | F1-Score | Confusion Matrix (TP / TN / FP / FN) | Provenance Breakdown |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Heuristic Judge** | **63.75%** | **0.923** | 0.300 | 0.453 | TP=12, TN=39, FP=1, FN=28 | Heuristic Rules (100%) |
| **LLM AutoRater (Qwen2.5)** | **45.00%** | 0.474 | **0.900** | **0.621** | TP=36, TN=0, **FP=40**, FN=4 | `qwen_local_inference` (80/80, 100%) |

*Oracle-Assisted Evaluation Disclosure*: In `scripts/evaluate_autorater.py` and `src/agentic_rag/prompts/sufficiency_v1.py`, the sufficiency judge prompt includes `PLANNED REQUIRED FACTS TO VERIFY`, which are supplied directly from the benchmark's curated metadata (`rec.get("required_facts")`). Consequently, this evaluation is **oracle-assisted** rather than fully autonomous: it assesses whether the model can verify context sufficiency when given human-curated target facts, not whether the model can autonomously determine what facts are necessary from scratch.

*Silver LLM Concordance Note*: `data/evaluation/independent_llm_annotations.jsonl` represents a static pre-annotation curation artifact from benchmark development (commit `fe536df`), showing 100.0% concordance with candidate labels. It is not a live evaluator.

### 3.3 Single-Class Aware Reasoning Breakdown
Previous metrics on single-class subsets generated misleading positive-class F1 scores. We corrected the breakdown to report true negative rate (TNR / specificity) on negative slices and true positive rate (TPR / sensitivity) on positive slices:

| Reasoning Type | Count | Accuracy | Slice Dist. | Primary Slice Metric | TP | TN | FP | FN | Failure Mode Analysis |
|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---|
| `single-hop-factual` | 31 | 96.77% | S:31 / I:0 | TPR (Sensitivity): **0.968** (30/31) | 30 | 0 | 0 | 1 | Correctly identifies factual presence. |
| `false-premise-detection` | 9 | 66.67% | S:9 / I:0 | TPR (Sensitivity): **0.667** (6/9) | 6 | 0 | 0 | 3 | Moderate capability to detect refutations. |
| `missing-intermediate-fact` | 19 | **0.00%** | S:0 / I:19 | TNR (Specificity): **0.000** (0/19) | 0 | 0 | 19 | 0 | **Sycophancy / Positive Bias**: Accepts birthplace/residence as occupation. |
| `entity-relation-mismatch` | 11 | **0.00%** | S:0 / I:11 | TNR (Specificity): **0.000** (0/11) | 0 | 0 | 11 | 0 | Ignores entity relation mismatches. |
| `unsubstantiated-claim` | 8 | **0.00%** | S:0 / I:8 | TNR (Specificity): **0.000** (0/8) | 0 | 0 | 8 | 0 | Assumes unstated facts without evidence. |
| `contradictory-evidence` | 1 | **0.00%** | S:0 / I:1 | TNR (Specificity): **0.000** (0/1) | 0 | 0 | 1 | 0 | Cannot resolve evidential conflict. |
| `partial-multi-attribute` | 1 | **0.00%** | S:0 / I:1 | TNR (Specificity): **0.000** (0/1) | 0 | 0 | 1 | 0 | Misses missing secondary attribute. |

### 3.4 Key Scientific Finding on LLM Autoraters
Small open models (0.5B parameters) suffer from severe **positivity bias / sycophancy** when acting as sufficiency judges without grounding constraints ($40/40$ false positives on negative instances). The model assumes any related topical context is sufficient. This highlights the scientific justification for EvidenceFirst's two-stage architecture: pairing LLM extraction with deterministic claim entailment verification.

---

## 4. Phase 3: Difficult-Query Retrieval Improvements

### 4.1 Methodology & Technical Constraints
Standard keyword retrieval struggles with domain-specific technical queries containing specialized acronyms (e.g. `TEVV`, `AI RMF`, `CSF`, `SSDF`, `GenAI`). We introduced:
1. `NISTTerminologyExpander`: Deterministic query expansion mapping domain abbreviations to canonical terminology.
2. `adaptive_score_fusion`: Min-max score normalization combining dense similarity and BM25 lexical matching with an adaptive co-occurrence bonus when chunks appear in both candidate lists.

*Technical Note on Dense Backend*: In this improved retrieval experiment, the dense-only baseline and dense fusion components operated using a deterministic **feature-hash fallback** (128-dimensional hashing vector representation indexed via FAISS) because the neural embedding model (`BAAI/bge-small-en-v1.5`) was unavailable in the local offline test environment. This particular dense result reflects the performance of the deterministic hashing fallback under offline execution, and must not be described as neural semantic retrieval or implied to have used a neural FAISS embedding backend.

### 4.2 Benchmark Results on NIST Corpus (54 Answerable Queries, 1,287 Chunks)

| Configuration | Recall@5 | Recall@10 | MRR | Median Latency (ms) | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| Dense-only (FAISS over feature-hash vectors) | 0.3704 | 0.5463 | 0.3090 | **0.7** | Feature-hash fallback (offline environment; neural embedder unavailable) |
| BM25-only (Okapi) | 0.7130 | 0.8333 | 0.5349 | 1.2 | Strong keyword match |
| Hybrid (RRF Fusion) | 0.6389 | 0.7870 | 0.4454 | 1.4 | Baseline rank fusion |
| Hybrid + Reranker (Cross-Encoder) | 0.7315 | **0.8611** | **0.6218** | 401.2 | Baseline reranker configuration |
| **Adaptive Score Fusion (Dense+BM25)** | 0.7222 | 0.8148 | 0.5124 | 1.7 | **+15.04% MRR over RRF without reranker** |
| **Adaptive Hybrid + Reranker (Improved)** | **0.7407** | 0.8519 | 0.6214 | 549.2 | **Peak Recall@5 (+1.26% over baseline)** |

*Retrieval Latency & Backend Provenance Caveat*:
The reported median retrieval latencies reflect full end-to-end retrieval times: first-stage retrieval (BM25 and feature-hash vector search) executes rapidly in 0.7–1.7 ms, whereas the two reranked configurations exhibit substantially higher latencies (401.2 ms and 549.2 ms). While local weights for `cross-encoder/ms-marco-MiniLM-L-6-v2` exist in `~/.cache/huggingface/hub/` and `CrossEncoderReranker()` initializes the neural model when SentenceTransformers is available, the saved evaluation artifacts (`results/retrieval_ablation_improved.csv` and `results/retrieval_improved_per_query.csv`) do not record a backend provenance field. Because execution latency alone cannot be used as substitute proof of neural execution, the saved artifacts cannot conclusively establish whether the recorded reranked runs executed neural CrossEncoder inference or lexical fallback, and backend provenance for this specific retrieval run must be qualified as unrecorded.

---

## 5. Phase 4: Multi-Step Cross-Document Recovery

### 5.1 Benchmark Scope & Anti-Leakage Protocol
- **Benchmark Scope**: Evaluated on a 13-chunk synthetic proof-of-concept corpus modeling multi-hop dependencies across independent documents (`technova_overview.txt`, `cybro_staff.txt`, `cybro_patents.txt`). The suite consists of six bounded evaluation questions (3 multi-hop chains, 1 single-hop query, and 2 unresolvable distractors).
- **Synthetic Proof-of-Concept Status**: The results demonstrate the algorithmic validity of sequential bridge-entity discovery, cycle prevention, and non-leakage mechanics. This 83.3% result is strictly a **six-case synthetic proof-of-concept result** and must not be interpreted as a generalizable real-world multi-hop benchmark finding.
- **Elimination of Downstream Leakage**: The recovery query reformulator extracts **only** the immediate missing entity from the currently retrieved evidence. Downstream entities (e.g. patent details, award locations) are completely hidden until intermediate chunks are discovered.
- **Safety Mechanisms**: Cycle detection prevents infinite query loops; duplicate chunk pruning ensures new information is gathered at each hop.

### 5.2 Depth Ablation Results ($K=0$ to $K=3$)

| Maximum Depth | Recovery Success Rate | Resolved Instances | Average Hops Executed | Avg New Chunks Gathered | Termination Behavior |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$K=0$** (No Recovery) | 16.7% | 1/6 | 0.00 | 0.00 | Baseline one-shot |
| **$K=1$** (Single-Step) | 33.3% | 2/6 | 0.83 | 1.33 | Resolves direct 1-hop dependencies |
| **$K=2$** (Two-Step) | 66.7% | 4/6 | 1.50 | 1.83 | Resolves 2-hop bridge chains |
| **$K=3$** (Three-Step Deep) | **83.3%** | **5/6** | 1.67 | 2.00 | Resolves full cross-document chains (synthetic POC suite) |

*Proof of Non-Leakage*: Unit tests in `tests/test_multistep_recovery.py::test_downstream_entity_unavailable_until_intermediate_discovery` prove that at $K=1$, the downstream entity `doc_c:c1` is absent from retrieved evidence, confirming that success at $K \ge 2$ is driven strictly by sequential discovery.

---

## 6. Phase 5: NIST End-to-End Comparative Evaluation

### 6.1 Experimental Setup
- **Corpus**: 1,287 chunks across 24 official NIST publications in `data/processed/`.
- **Query Set**: 64 total queries (54 answerable + 10 unanswerable queries targeting unstated years, budgets, or external documents).
- **Retrieval Parity**: All three systems (B0, B1, E1) use identical `mode="hybrid_rerank"` retrieval.
- **Framing Term Correction**: Meta-corpus framing terms (`nist`, `profile`, `report`, `publication`, `sp`, `ir`, `generative`, `genai`) were added to `QUESTION_FRAMING_TERMS` in `src/evidencefirst_rag/sufficiency.py`. This resolved false abstentions on definition questions where retrieved text answers the concept without repeating document framing words.

### 6.2 End-to-End Results Across 64 Queries

| System | Total Coverage | Answerable Coverage | Unsupported Rate (All Queries, $/64$) | Unsupported Rate (Answered, $/\text{Ans}$) | Unanswerable Abstention Accuracy | Median Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B0 Standard RAG** (Retrieve $\to$ Generate) | 100.0% (64/64) | 100.0% (54/54) | 15.6% (10/64) | 15.6% (10/64) | 0.0% (0/10) | 561.6 |
| **B1 Verify-Only RAG** (Retrieve $\to$ Gen $\to$ Verify) | 100.0% (64/64) | 100.0% (54/54) | 15.6% (10/64) | 15.6% (10/64) | 0.0% (0/10) | 508.6 |
| **E1 EvidenceFirst RAG** (Full Sufficiency Pipeline) | 70.3% (45/64) | **83.3% (45/54)** | **0.0% (0/64)** | **0.0% (0/45)** | **100.0% (10/10)** | 542.9 |

### 6.3 Mathematical Denominator Rigor & Benchmark Scope
- **Unsupported / Total ($/64$)**:
  $$\text{Rate}_{\text{total}} = \frac{\text{Unsupported Answers}}{\text{Total Queries}} = \frac{0}{64} = 0.0\% \quad (\text{vs } 10/64 = 15.6\% \text{ in B0})$$
- **Unsupported / Answered ($/\text{Answered}$)**:
  $$\text{Rate}_{\text{ans}} = \frac{\text{Unsupported Answers}}{\text{Answered Queries}} = \frac{0}{45} = 0.0\% \quad (\text{vs } 10/64 = 15.6\% \text{ in B0})$$
- **Unanswerable Abstention Accuracy**:
  $$\text{Abstention Rate} = \frac{\text{Abstained Unanswerable Queries}}{\text{Total Unanswerable Queries}} = \frac{10}{10} = 100.0\% \quad (\text{vs } 0.0\% \text{ in B0})$$

*Benchmark Scope & Scientific Caveats*:
1. **Unvalidated Reference Labels**: The reference answers, relevance spans, and sufficiency labels in this benchmark have not been independently human-validated; adjudicated human gold status remains strictly PENDING.
2. **Limited Negative Queries**: The evaluation set contains only 10 unanswerable queries ($N=10$ unanswerable vs. $N=54$ answerable).
3. **Benchmark-Specific Findings**: The system recorded zero unsupported answers against the current benchmark labels ($0.0\%$ unsupported rate). This finding must be interpreted strictly as benchmark-specific performance on this curated evaluation set, rather than an unconstrained real-world guarantee that all hallucinations are eliminated across arbitrary distributions.

---

## 7. How to Reproduce All Experiments

Reproduction requirements depend on local environment dependencies and model weights:
- **Offline Execution & Fallback Behavior**: Offline execution without network access requires locally cached model weights and relevant Python dependencies (HuggingFace Hub, PyTorch, Transformers, SentenceTransformers). When local neural weights are unavailable, loaders gracefully activate deterministic feature-hash and lexical reranking fallbacks, ensuring reproducible pipeline runs without failing or contacting remote hosts.
- **Cached vs. Fresh AutoRater Inference**: By default, `scripts/evaluate_autorater.py` reads pre-generated, verified model completions from `data/evaluation/.cache_autorater_qwen/`, which reproduces the 80-instance evaluation deterministically. Running fresh local model inference requires clearing or bypassing the cache (`--force-fresh`), a local HuggingFace cache containing `Qwen/Qwen2.5-0.5B-Instruct` weights, and PyTorch.

```bash
# Set offline environment variables
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 1. Human Validation Status and Package Export
python scripts/manage_human_validation.py status
python scripts/manage_human_validation.py export --name blinded_package

# 2. LLM AutoRater Evaluation (Provenance-Tracked)
python scripts/evaluate_autorater.py

# 3. Retrieval Improvements Evaluation (6 Modes on NIST)
python scripts/evaluate_retrieval_improvements.py

# 4. Multi-Step Cross-Document Recovery (K=0 to K=3)
python scripts/evaluate_multistep_recovery.py

# 5. NIST 64-Query End-to-End Comparative Pipeline
python scripts/evaluate_nist_end_to_end.py

# 6. Run Complete Test Suite (183 Unit & Regression Tests: 182 passed, 0 failures, 1 skipped)
python -m unittest discover -s tests -v

# 7. Validate Skill Packaging
python scripts/validate_skill.py
```
