# Human Annotation Guide: Sufficient Context Verification

This guide outlines the protocol for human verification of context sufficiency on the **EvidenceFirst Sufficient-Context Benchmark** subset:
`data/evaluation/human_verification_subset.jsonl`

---

## 1. Objective & The Diligent Reader Standard

The evaluation objective is to determine whether a given text context provides sufficient information for an attentive, careful reader to definitively answer the user's question.

We adopt the **Diligent Reader Principle** from Joren et al. (ICLR 2025):
> A context is **sufficient** if and only if a diligent reader, relying **strictly and solely on the provided text passage**, can construct a complete, unambiguous, and fully supported answer to the question.

---

## 2. Core Annotation Principles

Human annotators must strictly follow these eight rules:

1. **SUFFICIENT**  
   The context contains enough explicit evidence for a diligent reader to construct a definitive, unambiguous answer using **only** the supplied context.

2. **INSUFFICIENT**  
   The context does not contain enough evidence to construct a definitive answer.

3. **Multi-Hop Reasoning within Context is Allowed**  
   If answering requires connecting multiple sentences or facts that are *all present within the provided context*, that deduction is valid and should be considered supported.

4. **Unsupported Leaps of Faith are NOT Allowed**  
   You must not assume missing bridge facts, extrapolate beyond what is stated, or give the context the "benefit of the doubt" when an essential link is omitted.

5. **Do NOT Use Outside Knowledge or Web Search**  
   Evaluate the passage in complete isolation. Even if you know the true answer from your general knowledge, or even if the answer is a famous historical fact, if the provided context does not state or logically entail it, the label **must be INSUFFICIENT**.

6. **Do NOT Look at Candidate Labels or LLM Annotations**  
   This verification pass is strictly blinded. Do not inspect `candidate_label`, curator rationales, or `independent_llm_annotations.jsonl`. Your judgment must reflect independent human evaluation of the question and context alone.

7. **All Required Facts Must Be Supported**  
   If a question asks for multiple attributes (e.g., who founded a company and what year it was founded), **all required components** must be supported by the context. If any required attribute is missing, the label **must be INSUFFICIENT**.

8. **Relevant But Insufficient Information Must Be Labeled INSUFFICIENT**  
   Contexts that are topically related, discuss the entities in depth, or describe surrounding background without stating the specific fact requested must be labeled **INSUFFICIENT**. Merely mentioning an entity or keyword does not constitute evidence.

---

## 3. Subset Selection & Provenance Documentation

To provide a representative, balanced, and tractable evaluation set for human verification, a 40-record subset was constructed:

- **Target Datasets**: 4 diverse question distributions:
  - **PopQA** (Mallen et al., 2023)
  - **FreshQA** (Vu et al., 2023)
  - **Natural Questions** (Kwiatkowski et al., 2019)
  - **EntityQuestions** (Sciavolino et al., 2021)
- **Question Allocation**: Exactly 5 original source questions per dataset.
- **Variant Pairing**: For every selected question, **both controlled context variants** from `benchmark.jsonl` are included (one designed to test sufficiency, one designed to test insufficiency).
- **Total Records**: $4 \text{ datasets} \times 5 \text{ questions} \times 2 \text{ variants} = 40 \text{ records}$.

### Deterministic Selection Method & Seed
- **Selection Seed**: `42`
- **Method**:
  1. For each dataset, distinct original source questions are identified in their order of first appearance in `benchmark.jsonl`.
  2. A pseudorandom number generator initialized with `seed=42` (`random.Random(42)`) deterministically samples 5 questions per dataset:
     `sorted(random.Random(42).sample(range(10), 5))` $\rightarrow$ question indices `[0, 1, 4, 6, 9]`.
  3. For each sampled question, both paired context variants are extracted in their original sequence.
  4. All internal labels, curator expectations, and model predictions are completely stripped, producing a clean, blinded schema.

---

## 4. Record Schema (`human_verification_subset.jsonl`)

Each line in `data/evaluation/human_verification_subset.jsonl` contains strictly the following fields:

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

### Fields:
- `id` *(string)*: Unique benchmark identifier matching `benchmark.jsonl`.
- `source_dataset` *(string)*: Name of the originating source dataset.
- `original_question` *(string)*: The question to evaluate.
- `context` *(string)*: The provided text passage.
- `human_decision` *(string)*: Annotator enters `"SUFFICIENT"` or `"INSUFFICIENT"`. Initially empty string `""`.
- `human_rationale` *(string)*: Brief sentence explaining the annotator's decision (e.g., identifying the supporting sentence or naming the missing fact). Initially empty string `""`.

---

## 5. Annotation Step-by-Step Instructions

1. Open `data/evaluation/human_verification_subset.jsonl`.
2. For each line:
   a. Read the `original_question`. Identify what specific fact or facts are requested.
   b. Read the `context` carefully.
   c. Ask: *"Can I formulate a complete and definitive answer using ONLY the statements in this context?"*
   d. If yes:
      - Set `"human_decision": "SUFFICIENT"`
      - In `"human_rationale"`, note the context phrase or multi-hop deduction that proves the answer.
   e. If no:
      - Set `"human_decision": "INSUFFICIENT"`
      - In `"human_rationale"`, specify what fact is missing, ambiguous, or unproven.
3. Save the completed file.
