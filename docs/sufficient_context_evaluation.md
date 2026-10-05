# EvidenceFirst Independent Sufficient-Context Benchmark

Technical specification, methodology, and evaluation guide for the **EvidenceFirst Independent Sufficient-Context Benchmark**, inspired by the ICLR 2025 paper:

> **"Sufficient Context: A New Lens on Retrieval Augmented Generation Systems"**  
> *Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, Cyrus Rashtchian* (ICLR 2025)  
> [Conference Paper Link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) | [Official Repository](https://github.com/hljoren/sufficientcontext)

---

## 1. Motivation: The Retrieval-Generation Boundary

Standard Retrieval-Augmented Generation (RAG) benchmarks typically measure end-to-end answer correctness using metrics such as BLEU, ROUGE, or token F1 against a single ground-truth reference. While valuable, these end-to-end metrics fail to diagnose the root cause of failures at the critical interface between retrieval and synthesis:

1. **Hallucination under Insufficiency**: When retrieved passages lack necessary facts, standard generation models often produce fluent but ungrounded answers by relying on parametric training priors or making unsupported leaps.
2. **Over-Retrieval**: Adding more retrieved chunks often introduces distracting, outdated, or contradictory context without clarifying whether the core factual requirements are satisfied.
3. **Absence of Abstention**: Without an explicit context sufficiency detector, a RAG system cannot determine whether to trigger query rewriting for recovery or selectively abstain from answering.

The **Sufficient Context** paradigm formulated by Joren et al. (ICLR 2025) reframes this challenge: evaluate directly whether a `(question, context)` pair contains sufficient evidence for a *diligent reader* to answer the query definitively, before committing to generation.

---

## 2. Benchmark Attribution & Independence

This repository implements the **EvidenceFirst Independent Sufficient-Context Benchmark**:

- **Independent Benchmark Suite**: This benchmark was created independently as an open evaluation harness for the EvidenceFirst architecture. It does **not** claim to reproduce Google's internal Gemini 1.5 Pro autorater experiment, nor does it reuse the paper's 115 private/human-curated experimental instances.
- **Real Source Questions**: Questions and answers are obtained verbatim from four recognized public QA benchmark distributions (PopQA, FreshQA, Natural Questions, EntityQuestions) with genuine source record IDs.
- **Explicitly Constructed Evaluation Contexts**: For our independent benchmark, evaluation contexts are controlled passage variants constructed to test sufficiency detection. They are explicitly marked with `context_source_type: "constructed_controlled_context"` and are never claimed to be sampled from the original dataset texts.
- **Gold Label Integrity**: All records are initialized with `gold_label: "PENDING"`. Curator reference expectations are isolated under `candidate_label` and `rationale` to ensure unverified heuristics are never presented as human gold ground truth.

```
REAL SOURCE QUESTION (PopQA, FreshQA, Natural Questions, EntityQuestions)
        ↓
independent context construction/selection (controlled variants)
        ↓
QUESTION + CONTEXT (context_source_type: "constructed_controlled_context")
        ↓
independent human annotation (annotation_template.jsonl)
        ↓
GOLD SUFFICIENT / INSUFFICIENT (gold_label: "PENDING" until validated)
        ↓
evaluate our sufficiency detector (run_sufficiency_benchmark.py)
```

---

## 3. The Diligent Reader Standard

Context sufficiency is defined with respect to an idealized **diligent reader**:

### SUFFICIENT
A diligent reader can construct a definitive, unambiguous, and fully supported answer using **only** the supplied context, without bringing in external world knowledge. Multi-hop deductions are deemed sufficient if and only if **all intermediate linking facts** are explicitly stated within the context.

### INSUFFICIENT
A diligent reader cannot definitively answer the question based solely on the context. Insufficiency arises from several distinct failure modes:
- **Missing Intermediate Fact**: An essential bridge entity in a multi-hop inference is omitted.
- **Partial Multi-Attribute Coverage**: A multi-part question where only a subset of requested attributes is present (e.g., finding who founded a company but not the founding year).
- **Entity-Relation Mismatch**: The passage mentions the subject entity or answer type, but does not state the queried relation.
- **Temporal Obsolescence**: The context reflects a historical state that has since been superseded.
- **Contradictory Evidence**: Multiple statements within the context provide incompatible values for a required fact.
- **Unsubstantiated Claim / Mere Mention**: The answer entity appears as a token string without being attributed to the queried relationship.

> [!IMPORTANT]
> The source dataset's factual ground-truth answer (`source_qa_answer`) must be strictly distinguished from context sufficiency (`gold_label`). Even if the world-knowledge answer to a question is universally known, if the provided context lacks evidence, the context is `INSUFFICIENT`.

---

## 4. Public Source Dataset Pools & Provenance

The benchmark samples 10 real source questions from each of four public QA distributions, constructing two controlled context variants per question (one candidate `SUFFICIENT`, one candidate `INSUFFICIENT`), yielding **80 benchmark records**:

| Dataset Source | Reference | Repository & Source File | Provenance & Source IDs |
|---|---|---|---|
| **PopQA** | Mallen et al., 2023 | [AlexTMallen/adaptive-retrieval](https://github.com/AlexTMallen/adaptive-retrieval)<br>`data/popQA.tsv` | Genuine Wikidata entity IDs (`source_record_id: "4222362"`). Raw cached sample in `data/evaluation/sources/popqa_raw_sample.tsv`. Local locator tracked in `source_locator`. |
| **FreshQA** | Vu et al., 2023 | [freshllms/freshqa](https://github.com/freshllms/freshqa)<br>Official weekly Google Sheet | Genuine FreshQA integer row IDs (`source_record_id: "0"`). Evaluates false-premise and dynamic world knowledge questions. Raw cached sample in `data/evaluation/sources/freshqa_raw_sample.csv`. Local locator tracked in `source_locator`. |
| **Natural Questions** | Kwiatkowski et al., 2019 | [google-research-datasets/natural-questions](https://github.com/google-research-datasets/natural-questions)<br>`nq_open/NQ-open.dev.jsonl` | Real NQ dev queries with open-domain answer spans. Upstream dataset lacks record IDs (`source_record_id: null`). Local tracking locator preserved in `source_locator: "nq-dev-001"`. Raw cached sample in `data/evaluation/sources/nq_open_raw_sample.jsonl`. |
| **EntityQuestions** | Sciavolino et al., 2021 | [princeton-nlp/EntityQuestions](https://github.com/princeton-nlp/EntityQuestions)<br>`dataset/dev/P19.dev.json`, `P106.dev.json` | Real entity-relation queries across Wikidata relations. Upstream dataset lacks record IDs (`source_record_id: null`). Local tracking locator preserved in `source_locator: "eq-P19-001"`. Raw cached sample in `data/evaluation/sources/`. |

---

## 5. Record Schema (`data/evaluation/benchmark.jsonl`)

```json
{
  "id": "ef-popqa-001",
  "source_dataset": "PopQA",
  "source_record_id": "4222362",
  "source_locator": "4222362",
  "original_question": "What is George Rankin's occupation?",
  "source_qa_answer": "politician",
  "context_source_type": "constructed_controlled_context",
  "context": "George Rankin was an Australian politician and soldier who served as a member of both the Australian House of Representatives and the Senate.",
  "gold_label": "PENDING",
  "candidate_label": "SUFFICIENT",
  "required_facts": [
    "occupation of George Rankin"
  ],
  "reasoning_type": "single-hop-factual",
  "rationale": "The context directly identifies George Rankin's occupation as an Australian politician. A diligent reader requires no outside information.",
  "annotator_id": null,
  "annotation_notes": ""
}
```

### Annotation Workflow (`data/evaluation/annotation_template.jsonl`)

To ensure scientific independence, human annotation is conducted as follows:

1. Annotators open `data/evaluation/annotation_template.jsonl`.
2. For each record, the annotator inspects `original_question` and `context` exclusively under the **diligent reader rule**.
3. Annotators must not consult the internet or use prior knowledge about the entities.
4. The annotator records:
   - `gold_label`: `"SUFFICIENT"` or `"INSUFFICIENT"`
   - `annotator_id`: Reviewer identifier
   - `annotation_timestamp`: ISO 8601 timestamp
   - `annotator_notes`: Optional comments on borderline phrasing or nuances
5. Verified records are saved to `benchmark.jsonl` or passed directly to `run_sufficiency_benchmark.py --benchmark path/to/annotated.jsonl`.

---

## 6. Three-Tier Label Taxonomy & Independent LLM Annotations

To ensure scientific rigor, avoid data leakage, and maintain complete transparency, the benchmark strictly enforces a **three-tier label taxonomy**:

1. **`candidate_label` = curator/development expectation**  
   - Established during dataset curation and passage construction.  
   - Serves as an internal heuristic for development, sanity checks, and pipeline debugging.  
   - Must **not** be presented as independent human ground truth or used to claim benchmark accuracy.

2. **`independent_annotation` = separate LLM-assisted silver annotation**  
   - Stored in `data/evaluation/independent_llm_annotations.jsonl`.  
   - Generated by an independent LLM evaluating the binary diligent-reader question without access to `candidate_label`, curator rationales, or benchmark statistics.  
   - It is an LLM-assisted silver annotation artifact. It is **NOT human gold**, it is **NOT model evaluation ground truth**, and it **must not be described as 100% accurate**.  
   - Serves as an intermediate consistency check and an assistive pre-annotation artifact for human reviewers.

3. **`gold_label` = human-verified ground truth, currently PENDING**  
   - The authoritative reference standard for model evaluation.  
   - All 80 records in `data/evaluation/benchmark.jsonl` have `gold_label: "PENDING"`.  
   - Gold labels remain **PENDING** until independently verified human annotation is performed.

### Independent LLM Annotation Artifact (`independent_llm_annotations.jsonl`)

The artifact `data/evaluation/independent_llm_annotations.jsonl` contains 80 records conforming to:
```json
{
  "id": "ef-popqa-001",
  "independent_annotation": "SUFFICIENT",
  "missing_facts": "",
  "annotation_rationale": "Context explicitly states that George Rankin was an Australian politician and soldier.",
  "annotator_type": "LLM_independent"
}
```
- **Strict Isolation**: Contains zero references to `candidate_label`, `curator_rationale`, `rationale`, or `gold_label`.
- **Distribution Sanity Check**: Contains 40 `SUFFICIENT` and 40 `INSUFFICIENT` annotations. This is an annotation distribution balance sanity check on the curated dataset, **not** model performance.

---

## 7. EvidenceFirst Detector Architecture

The EvidenceFirst sufficiency evaluation utilizes three core components from `src/agentic_rag`:

1. **Planner (`GeneralPlanner`)**:
   - Parses the question into discrete `RequiredFact` elements with `priority=MUST`.
   - Identifies multi-part questions connected by conjunctions ("and", "also", "as well as") and extracts key content-bearing terms.
2. **Drafter (`GenericExtractiveDrafter`)**:
   - Matches candidate evidence snippets against the planned required facts to form grounded claims.
3. **Sufficiency Judge (`AutoraterStyleSufficiencyJudge`)**:
   - Implements the autorater verification pattern: verifies that every must-have required fact is covered by supporting context snippets.
   - Flags missing facts, unsupported claims, and conflicting evidence groups.
   - Computes a sufficiency score ($S \in [0.0, 1.0]$) and assigns an `AnswerabilityLabel` (`SUFFICIENT`, `INSUFFICIENT`, `CONFLICTING`, `UNANSWERABLE`).

---

## 8. Empirical Evaluation Results

Running the evaluation runner against the 80 benchmark instances yields the following results:

### Annotation Status
- Total Instances: 80
- Verified Human Gold Labels: 0 (80 `PENDING`)

### Candidate Alignment (Development Heuristic)
When evaluated against curator candidate expectations, the zero-shot planner achieves:

- **Overall Alignment**: 63.7% (51/80) in zero-shot planner mode.
- **Candidate Precision on SUFFICIENT**: 0.923.
- **True Negative Recognition**: 39/40 (97.5% on candidate insufficient contexts).
- **Reasoning Type Breakdown**:
  - `contradictory-evidence`: 100.0% agreement
  - `entity-relation-mismatch`: 100.0% agreement
  - `unsubstantiated-claim`: 100.0% agreement
  - `partial-multi-attribute`: 100.0% agreement
  - `missing-intermediate-fact`: 94.7% agreement
  - `false-premise-detection`: 44.4% agreement
  - `single-hop-factual`: 25.8% agreement (lexical variance between question terms and passage phrasing)

### Key Observation: The Lexical-Semantic Gap
In instances like `ef-popqa-001` ("What is George Rankin's occupation?"), the passage states: *"George Rankin was an Australian politician..."*. The deterministic lexical judge checks for the word *"occupation"*, which is absent from the passage even though *"politician"* directly answers the question. This directly illustrates the fundamental challenge formulated in Joren et al. (2025): **deterministic lexical verification is highly conservative against false positives, but exposes the boundary where semantic autorater reasoning is needed to bridge natural language variation.**

---

## 9. Comparison: Benchmark vs. B0/E1 Functional Evaluation

The EvidenceFirst repository provides two complementary evaluation tools:

| Evaluation Dimension | B0 vs. E1 Controlled Functional Harness (`examples/run_evaluation.py`) | EvidenceFirst Sufficient-Context Benchmark (`examples/run_sufficiency_benchmark.py`) |
|---|---|---|
| **Primary Focus** | End-to-end multi-corpus RAG pipeline comparison (Baseline vs. EvidenceFirst). | Isolated context sufficiency detection at the context boundary. |
| **Data Scope** | NovaTech enterprise scenario (company, history, hr, products corpora). | 80 instances across 4 public QA datasets (PopQA, FreshQA, NQ, EntityQuestions). |
| **Key Metrics** | Retrieval recovery rate, fact grounding score, selective abstention rate, token citation precision. | Context sufficiency accuracy, precision, recall, F1, confusion matrix, annotation readiness. |
| **Recovery Loop** | Exercises multi-turn query rewriting and follow-up corpus fetching. | Evaluates single-step sufficiency decision on fixed `(question, context)` pairs. |

---

## 10. Blinded Human-Verification Subset (`human_verification_subset.jsonl`)

To facilitate independent, unbiased human verification, a blinded 40-record subset has been prepared with an accompanying protocol guide:
- **Data File**: `data/evaluation/human_verification_subset.jsonl`
- **Protocol Guide**: [`data/evaluation/HUMAN_ANNOTATION_GUIDE.md`](file:///Users/kanikagadiya/Documents/projects/evidencefirst/evidencefirst-demo/data/evaluation/HUMAN_ANNOTATION_GUIDE.md)
- **Generator Utility**: `examples/prepare_human_verification_subset.py`

### Deterministic Selection Method & Seed:
- **Seed**: `42`
- **Method**: For each of the 4 public datasets (PopQA, FreshQA, Natural Questions, EntityQuestions), original source questions are ordered by initial appearance in `benchmark.jsonl`. A pseudo-random generator initialized with `seed=42` (`random.Random(42)`) samples 5 questions per dataset (indices `[0, 1, 4, 6, 9]`). For each selected question, both paired context variants are extracted in their original sequence, yielding 10 records per dataset and **40 records total**.

### Critical Blinding Requirement:
The human-facing file is strictly blinded and contains **only** the following 6 fields:
```json
{
  "id": "ef-popqa-001",
  "source_dataset": "PopQA",
  "original_question": "What is George Rankin's occupation?",
  "context": "George Rankin was an Australian politician and soldier who served as a member of both the Australian House of Representatives and the Senate.",
  "human_decision": "",
  "human_rationale": ""
}
```
`human_decision` and `human_rationale` are initially empty strings (`""`). The file strictly omits `candidate_label`, `independent_annotation`, `curator_rationale`, and `gold_label`.

---

## 11. Execution Guide

### 1. Benchmark Data Preparation and Validation
```bash
python3 examples/prepare_sufficient_context_benchmark.py
```
Validates JSON schema, verifies record counts, and checks label distributions.

### 2. Benchmark Sufficiency Evaluation
```bash
# Zero-shot GeneralPlanner mode
python3 examples/run_sufficiency_benchmark.py

# Fact-grounded mode
python3 examples/run_sufficiency_benchmark.py --mode fact_grounded

# Verbose mode (full traces for all 80 examples)
python3 examples/run_sufficiency_benchmark.py --verbose

# Export results to JSON
python3 examples/run_sufficiency_benchmark.py --output-json results.json
```

### 3. Automated Unit Testing
```bash
# Benchmark-specific tests
python3 -m unittest tests/test_sufficient_context_benchmark.py -v

# Full project test suite (all 65 tests)
python3 -m unittest discover -s tests -v

# Skill package validation
python3 scripts/validate_skill.py
```

### 4. Legacy Draft Candidate Preservation
The initial draft candidate set has been preserved intact at:
`data/evaluation/legacy_draft_candidates.jsonl`
This ensures all earlier exploratory questions and rationales remain available for archival reference.
