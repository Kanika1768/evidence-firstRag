# EvidenceFirst RAG: Human Annotation Protocol & Sufficiency Guidelines

## 1. Objective and Theoretical Standard

This protocol governs the human annotation and gold-standard verification for the **EvidenceFirst Context Sufficiency Benchmark** (105 evaluation questions across factoid, multi-hop, and unanswerable scenarios).

The benchmark operationalizes the **Diligent Reader Standard** formulated in *Joren et al. (ICLR 2025)*:

> *"A context is sufficient if and only if a diligent reader, possessing general language comprehension but zero external domain knowledge, can construct a complete, definitive, and verifiable answer to the query using strictly the text provided."*

---

## 2. Annotation Task Formulation

Given a tuple `(Question, Context, Candidate Answer)`:
The annotator must independently assign one of two binary sufficiency labels:

- **`SUFFICIENT`**: The provided context contains all necessary facts and logical links required to construct an indisputable answer to the question.
- **`INSUFFICIENT`**: The provided context lacks one or more critical facts, contains irreconcilable contradictions, or fails to substantiate the required entities/relations.

### Key Rules & Constraints:
1. **Zero External Knowledge**: Annotators must NOT assume or import outside facts (e.g., historical dates, company headquarters, scientific constants) not explicitly stated in the context.
2. **Valid Intra-Context Deductions**: Multi-hop logical deduction *within* the provided snippets is permitted and encouraged (e.g., if Sentence A states *"X acquired Y in 2020"* and Sentence B states *"Y was founded by Z"*, the reader may deduce that X acquired a company founded by Z).
3. **No Speculative Leaps**: Missing intermediate links, ambiguous pronouns without antecedents, or ungrounded temporal assumptions render the context `INSUFFICIENT`.
4. **False Premise Handling**: If a question presupposes a false premise (e.g., *"What is the name of the first animal to land on the Moon?"*), the context is `SUFFICIENT` only if it explicitly refutes or clarifies the premise (e.g., *"No non-human animal has ever landed on the Moon"*).

---

## 3. Question Categories & Benchmark Distribution

The evaluation benchmark (`data/eval/annotated_questions.csv`) consists of **105 curated items** balanced across:

| Question Type | Target Ratio | Count | Description |
|:---|:---:|:---:|:---|
| **Factoid (Single-Hop)** | ~40% | 42 | Direct entity, date, location, or numerical attribute questions. |
| **Multi-Hop / Compositional** | ~30% | 31 | Questions requiring synthesis of 2+ disjoint facts or documents. |
| **Unanswerable / Adversarial** | ~30% | 32 | Questions where crucial facts are deliberately absent or masked. |
| **Total** | 100% | 105 | Balanced: 52 Sufficient (~49.5%) / 53 Insufficient (~50.5%) |

---

## 4. Multi-Stage Annotation Workflow

### Stage 1: Initial Silver LLM Annotations
- All 105 records receive initial automated silver labels using structured prompting (`agentic_rag.prompts.sufficiency_v1`).
- The LLM judge outputs `sufficient: bool`, `probability: float`, `missing_information: list[str]`, and `rationale: str`.
- Silver labels serve solely as an initial baseline and are never conflated with verified human ground truth.

### Stage 2: Independent Human Review & Blinding
- Human annotators inspect the query and context in a strictly blinded setting.
- Annotators are shielded from:
  - System candidate labels
  - Model confidence scores
  - Intermediate pipeline traces

### Stage 3: Double-Annotation Protocol (20–30% Subset)
- Exactly **25 records (23.8%)** are assigned to two independent annotators (Annotator A and Annotator B).
- The double-annotation subset is stratified across question types:
  - 10 Factoid instances
  - 8 Multi-hop instances
  - 7 Unanswerable instances
- Both annotators record their labels without cross-communication.

### Stage 4: Disagreement Resolution & Inter-Annotator Agreement
- Disagreements between Annotator A and Annotator B are flagged for adjudication.
- An adjudicator convenes an alignment discussion to review the text against the Diligent Reader criteria.
- **Inter-Annotator Metrics Recorded**:
  - Raw Percentage Agreement: $P_o = \frac{\text{Agreed Instances}}{\text{Total Double-Annotated}}$
  - Cohen's Kappa: $\kappa = \frac{P_o - P_e}{1 - P_e}$
  - Target agreement threshold: $\kappa \ge 0.85$ (strong inter-rater reliability).

### Stage 5: Gold Standard Finalization
- Ground truth `gold_label` remains strictly marked **`PENDING`** until human verification is completed and signed off.
- Automated tests enforce that no synthetic or unverified labels overwrite `gold_label`.

---

## 5. Decision Rubric & Concrete Examples

### Example 1: Sufficient Multi-Hop
- **Question**: *"Who founded NovaTech and when was it founded?"*
- **Context**:
  - `[company.txt]`: *"NovaTech is a software company founded by Priya Mehta."*
  - `[history.txt]`: *"NovaTech was founded in 2019 in Bengaluru."*
- **Label**: **`SUFFICIENT`**
- **Rationale**: Both required facts (founder = Priya Mehta, year = 2019) are explicitly present across the retrieved snippets.

### Example 2: Insufficient Partial Evidence
- **Question**: *"Who founded NovaTech and when was it founded?"*
- **Context**:
  - `[company.txt]`: *"NovaTech is a software company founded by Priya Mehta."*
- **Label**: **`INSUFFICIENT`**
- **Missing Facts**: `["Founding year of NovaTech"]`
- **Rationale**: The founder is documented, but the founding year is missing. Answering would require external knowledge or fabrication.

### Example 3: Unanswerable Entity Attribute
- **Question**: *"Who is the Chief Financial Officer (CFO) of NovaTech?"*
- **Context**:
  - `[company.txt]`: *"NovaTech was founded by Priya Mehta (CEO) and Rahul Sharma (CTO)."*
- **Label**: **`INSUFFICIENT`**
- **Missing Facts**: `["CFO of NovaTech"]`
- **Rationale**: The document details CEO and CTO but contains zero mentions of a CFO. Diligent reader must abstain.
