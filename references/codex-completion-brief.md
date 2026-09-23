# Codex Completion Brief

## Target architecture

```text
src/agentic_rag/
+-- contracts.py       # Stable dataclasses and Protocols
+-- orchestrator.py    # Agentic loop
+-- sufficiency.py     # Sufficient-context autorater and abstention policy
+-- evaluation.py      # FRAMES-style fixtures and metrics
+-- adapters/
|   +-- llm.py         # TODO-CODEX: provider client + structured output
|   +-- retriever.py   # TODO-CODEX: vector/BM25/SQL retriever adapters
|   +-- google_vertex_rag.py  # TODO-CODEX: native Google Cross-Corpus Retrieval mode
```

## TODO-CODEX order

1. **Contracts**
   - Review `contracts.py`.
   - Add fields only if required for the selected backend.
   - Current scaffold includes claim citations, missing-fact feedback queries, iteration traces, `sufficiency_score`, and conflict evidence groups.

2. **In-memory deterministic backend**
   - Implement a local fake retriever and fake LLM for tests.
   - This proves loop semantics without network calls.
   - Current scaffold provides dependency-free deterministic components in `src/agentic_rag/adapters/in_memory.py`.

3. **Orchestrator loop**
   - Maintain iteration state.
   - Merge snippets across iterations.
   - Stop on sufficient/unanswerable/max iterations.
   - Return partial answer if insufficient after budget.
   - Current scaffold enforces this portable loop in `src/agentic_rag/orchestrator.py`.

4. **LLM structured outputs** (`RDD-T-00000007`)
   - Convert prompts in `prompts-and-schemas.md` to provider-specific schemas.
   - Validate outputs before using them.
   - Retry malformed JSON once with a repair instruction.
   - Current scaffold includes dependency-free schema registry, strict JSON parsing, one-shot repair, and dataclass conversion helpers in `src/agentic_rag/adapters/llm.py`.
   - Implement in order:
     - `RDD-T-00000014`: Schema registry and dataclass conversion helpers. Completed.
     - `RDD-T-00000015`: Structured JSON parser and validation errors. Completed.
     - `RDD-T-00000016`: One-shot JSON repair protocol contract. Completed.

5. **Retriever adapters** (`RDD-T-00000008`)
   - Add at least one real adapter.
   - Preserve snippet provenance.
   - Include corpus id and document id for every hit.
   - Current scaffold includes a dependency-free lexical retriever in `src/agentic_rag/adapters/retriever.py`.
   - Lexical scoring is deterministic: unique overlapping query terms plus an exact-phrase bonus, then provenance-key tie-breaking.
   - Implement in order:
     - `RDD-T-00000017`: Provenance-preserving lexical retriever adapter. Completed.
     - `RDD-T-00000018`: Retrieval scoring and deduplication tests. Completed.

6. **Sufficient Context Judge** (`RDD-T-00000009`)
   - It must inspect original question, plan, snippets, and draft.
   - It must return missing facts and feedback queries when insufficient.
   - Add answerability categories and selective abstention behavior inspired by the Sufficient Context paper.
   - Current contracts include `AnswerabilityLabel` while preserving the existing `ContextAssessment.status` API.
   - Current sufficiency utilities include `AutoraterStyleSufficiencyJudge` for deterministic required-fact coverage, missing facts, unsupported claims, feedback queries, conflict evidence grouping, and unanswerable classification.
   - Current orchestrator guard applies selective abstention so insufficient contexts cannot remain fully answered.
   - Implement in order:
     - `RDD-T-00000019`: Sufficient Context answerability categories. Completed.
     - `RDD-T-00000020`: Autorater-style sufficiency judge. Completed.
     - `RDD-T-00000021`: Selective generation abstention policy. Completed.

7. **Evaluation** (`RDD-T-00000010`)
   - Add a tiny multi-hop fixture with at least two corpora and one distractor corpus.
   - Verify that routing does not search all corpora unless justified.
   - Verify that one missing fact triggers a targeted follow-up query.
   - Report fact coverage, fetch coverage, reasoning correctness, citation completeness, and iteration count.
   - Current evaluation utilities include fixture dataclasses, dependency-free metrics, and baseline-vs-candidate comparison for fact, fetch, reasoning, citation, and iteration reporting.
   - Implement in order:
     - `RDD-T-00000022`: FRAMES-style fixture format and metrics. Completed.
     - `RDD-T-00000023`: Iterative-vs-single-shot evaluation tests. Completed.

8. **Conflict-aware synthesis** (`RDD-T-00000011`)
   - Surface contradictory snippets instead of silently merging them.
   - Cite both sides of a conflict.
   - Return partial or conflict-aware status when a required fact has incompatible evidence.
   - Current contracts preserve conflict evidence groups in `ContextAssessment`, structured-output conversion, and `GroundedAnswer`.
   - Current in-memory synthesis cites each incompatible evidence group separately and returns a partial answer instead of merging conflicting facts.
   - Implement in order:
     - `RDD-T-00000024`: Conflict evidence contracts. Completed.
     - `RDD-T-00000025`: Conflict-aware judge and synthesis behavior. Completed.

9. **Native Google mode** (`RDD-T-00000012`)
   - If using Gemini Enterprise Agent Platform RAG Engine, add an adapter that calls Cross-Corpus Retrieval APIs.
   - Respect location, IAM, project, and corpus-resource requirements.
   - Keep Google-specific dependencies out of module import time.
   - Implement in order:
     - `RDD-T-00000026`: Google native mode configuration validation.
     - `RDD-T-00000027`: Google Cross Corpus request adapter seam.
     - `RDD-T-00000028`: Portable mode versus Google native mode documentation.

## Acceptance tests to add

- Query requiring two facts in two corpora succeeds after two retrieval iterations.
- Query with no available evidence returns `partial` or `unanswerable` and lists missing facts.
- Draft with an unsupported claim fails sufficiency.
- Conflicting snippets are surfaced rather than silently merged.
- All final claims have snippet ids.
