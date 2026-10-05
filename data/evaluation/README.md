# EvidenceFirst Independent Sufficient-Context Benchmark

The **EvidenceFirst Independent Sufficient-Context Benchmark** evaluates whether a retrieval-augmented generation (RAG) system can reliably assess context sufficiency at the retrieval-generation boundary.

Inspired by the ICLR 2025 paper:
> **"Sufficient Context: A New Lens on Retrieval Augmented Generation Systems"**  
> *Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, Cyrus Rashtchian* (ICLR 2025)  
> [Conference Paper Link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html) | [Official Repository](https://github.com/hljoren/sufficientcontext)

---

## 1. Important Scope & Independence Statement

- **Independent Benchmark Suite**: This dataset and evaluation harness is an independent benchmark developed for the EvidenceFirst project. It is **not** Google's internal autorater implementation and does **not** claim to reproduce the paper's original 115 human-labeled experimental instances or private annotations.
- **Controlled Context Methodology**: For our independent benchmark, questions are sampled verbatim from verified public datasets, while contexts are controlled passage variants constructed specifically to test sufficiency detection. These contexts are explicitly labeled as `context_source_type: "constructed_controlled_context"` and are **not** claimed to be directly sampled from the original dataset texts.
- **Gold Label Integrity**: In strict adherence to scientific rigor, all records are initialized with `gold_label: "PENDING"`. Curator reference expectations are isolated under `candidate_label` and `rationale` to ensure unverified heuristic expectations are never presented as human gold ground truth.

---

## 2. Benchmark Construction Workflow

The benchmark follows an explicit multi-stage construction pipeline:

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

## 3. Public Source Datasets & Real Source Questions

The benchmark samples 10 real source questions from each of four public QA distributions, constructing two controlled context variants per question (one candidate `SUFFICIENT`, one candidate `INSUFFICIENT`), yielding **80 benchmark records**:

| Dataset Source | Reference | Repository & File | Provenance & ID Scheme |
|---|---|---|---|
| **PopQA** | Mallen et al., 2023 | [AlexTMallen/adaptive-retrieval](https://github.com/AlexTMallen/adaptive-retrieval)<br>`data/popQA.tsv` | Genuine Wikidata entity IDs (`source_record_id: "4222362"`). Raw cached sample in `data/evaluation/sources/popqa_raw_sample.tsv`. Local locator tracked in `source_locator`. |
| **FreshQA** | Vu et al., 2023 | [freshllms/freshqa](https://github.com/freshllms/freshqa)<br>Official weekly Google Sheet | Genuine FreshQA integer row IDs (`source_record_id: "0"`). Evaluates false-premise and dynamic world knowledge questions. Raw cached sample in `data/evaluation/sources/freshqa_raw_sample.csv`. Local locator tracked in `source_locator`. |
| **Natural Questions** | Kwiatkowski et al., 2019 | [google-research-datasets/natural-questions](https://github.com/google-research-datasets/natural-questions)<br>`nq_open/NQ-open.dev.jsonl` | Real NQ dev queries with open-domain answer spans. Upstream dataset lacks record IDs (`source_record_id: null`). Local tracking locator preserved in `source_locator: "nq-dev-001"`. Raw cached sample in `data/evaluation/sources/nq_open_raw_sample.jsonl`. |
| **EntityQuestions** | Sciavolino et al., 2021 | [princeton-nlp/EntityQuestions](https://github.com/princeton-nlp/EntityQuestions)<br>`dataset/dev/P19.dev.json`, `P106.dev.json` | Real entity-relation queries across Wikidata relations. Upstream dataset lacks record IDs (`source_record_id: null`). Local tracking locator preserved in `source_locator: "eq-P19-001"`. Raw cached sample in `data/evaluation/sources/`. |

---

## 4. The Diligent Reader Standard

Under the ICLR 2025 formulation, context sufficiency is defined with respect to a hypothetical **diligent reader**:

- **SUFFICIENT**: A diligent reader can formulate a complete, unambiguous, and fully supported answer using **only** the supplied context, without relying on unstated background knowledge. Multi-hop deductions are deemed sufficient if and only if **all intermediate linking facts** are present within the context.
- **INSUFFICIENT**: A diligent reader cannot definitively answer the question using only the context. This includes:
  - Missing critical facts or missing intermediate links in a multi-hop chain.
  - Partial multi-attribute coverage (e.g., finding the founder but missing the founding year).
  - Entity-relation mismatches or ambiguity.
  - Temporal obsolescence (outdated context for time-sensitive questions).
  - Contradictory evidence across retrieved snippets.
  - Incidental mention of the answer string without factual attribution to the query subject.

> [!IMPORTANT]
> The source dataset's factual ground-truth answer (`source_qa_answer`) is **conceptually distinct** from context sufficiency (`gold_label`). Even if the world-knowledge answer to a question is known, if the provided context lacks evidence, the context is `INSUFFICIENT`.

---

## 5. Benchmark Record Schema (`benchmark.jsonl`)

Each line in `benchmark.jsonl` is a JSON object with the following schema:

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

### Field Definitions:
- `id` *(string, required)*: Unique record identifier (`ef-popqa-001` .. `ef-entityq-020`).
- `source_dataset` *(string, required)*: One of `"PopQA"`, `"FreshQA"`, `"Natural Questions"`, `"EntityQuestions"`.
- `source_record_id` *(string or null, required)*: Verified, genuine identifier from the source dataset for PopQA and FreshQA; `null` for Natural Questions and EntityQuestions because their upstream distributions lack record IDs.
- `source_locator` *(string, required)*: Tracking locator preserving reference indices across all sources (`"4222362"`, `"0"`, `"nq-dev-001"`, `"eq-P19-001"`).
- `original_question` *(string, required)*: The real input question taken verbatim from the public dataset.
- `source_qa_answer` *(string, required)*: Ground-truth answer string from the public source dataset.
- `context_source_type` *(string, required)*: Set to `"constructed_controlled_context"` to explicitly declare that the context is a controlled evaluation passage rather than original dataset text.
- `context` *(string, required)*: The text passage provided to the reader/model.
- `gold_label` *(string, required)*: `"PENDING"` until independently verified by human annotators.
- `candidate_label` *(string, required)*: Curated reference expectation (`"SUFFICIENT"` or `"INSUFFICIENT"`).
- `required_facts` *(list of strings, required)*: Atomic facts necessary to answer the question.
- `reasoning_type` *(string, required)*: Category of reasoning required.
- `rationale` *(string, required)*: Explanation justifying the candidate/gold label.
- `annotator_id` *(string or null)*: Identifier of the human judge who validated the record.
- `annotation_notes` *(string)*: Review notes or edge case comments.

---

## 6. Three-Tier Label Taxonomy & Independent LLM Annotations

To ensure scientific rigor, avoid data leakage, and maintain complete transparency, the EvidenceFirst benchmark strictly enforces a **three-tier label taxonomy**:

1. **`candidate_label` = curator/development expectation**  
   - Established during dataset curation and passage construction.  
   - Serves as an internal heuristic for development, sanity checks, and pipeline debugging.  
   - Must **not** be presented as independent human ground truth or used to claim benchmark accuracy.

2. **`independent_annotation` = separate LLM-assisted silver annotation**  
   - Stored in `data/evaluation/independent_llm_annotations.jsonl`.  
   - Generated by an independent LLM evaluating the binary diligent-reader question without access to `candidate_label`, curator rationales, or benchmark statistics.  
   - It is an LLM-assisted silver annotation artifact. It is **NOT human gold**, it is **NOT model evaluation ground truth**, and it **must not be described as 100% accurate**.  
   - Serves as a useful secondary check and an assistive pre-annotation artifact for human reviewers.

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

## 7. Legacy Draft Preservation

The initial draft candidate set has been preserved intact at:
`data/evaluation/legacy_draft_candidates.jsonl`

This guarantees that prior work, candidate questions, and handcrafted rationales remain accessible for historical review and research continuity.

---

## 8. Annotation Workflow (`annotation_template.jsonl`)

To conduct human gold-label annotation:

1. Open `data/evaluation/annotation_template.jsonl`.
2. For each record, read `original_question` and `context` strictly under the **diligent reader rule**.
3. Do **not** consult the internet or apply external world knowledge.
4. Set `gold_label` to `"SUFFICIENT"` if all required facts are explicitly supported; otherwise set to `"INSUFFICIENT"`.
5. Enter your `annotator_id` and optional `annotation_notes`.
6. Replace `benchmark.jsonl` with your verified records (or run `run_sufficiency_benchmark.py --benchmark path/to/annotated.jsonl`).

---

## 9. Blinded Human-Verification Subset (`human_verification_subset.jsonl`)

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

## 10. Execution Commands

### Prepare and validate benchmark data:
```bash
python3 examples/prepare_sufficient_context_benchmark.py
```

### Run the sufficiency detector evaluation:
```bash
python3 examples/run_sufficiency_benchmark.py
```

### Run automated unit tests:
```bash
python3 -m unittest tests/test_sufficient_context_benchmark.py -v
```

---

## 11. Benchmark Roadmap & Expansion

- **Current Verified Benchmark Size**: 80 instances (20 instances from each of 4 public sources: PopQA, FreshQA, Natural Questions, EntityQuestions).
- **Human Annotation Status**: All 80 records currently maintain `gold_label: "PENDING"`. Silver reference labels are provided by `independent_llm_annotations.jsonl`, and a 40-record blinded verification subset is prepared in `human_verification_subset.jsonl`.
- **Planned Future Expansion**: Expansion to **100–120 instances** is planned as future work, incorporating multi-hop retrieval challenges from HotpotQA / 2WikiMultiHopQA to evaluate complex multi-hop dependency chains.

