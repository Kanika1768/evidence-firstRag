# Agentic RAG Behavior Model

This is an implementation guide for the public behavior pattern described by Google Research and Google Cloud docs. It is not a copy of Google's internal system.

## Why standard RAG fails here

Standard RAG is often a single retrieval pass followed by generation. It fails on multi-hop questions where the first retrieved document only reveals an intermediate identifier, entity, date, or pointer that must be used in a second search. It also fails when relevant evidence is split across organizational data islands.

## Main roles

| Role | Responsibility | Output |
|---|---|---|
| Orchestrator / Router | Owns the query lifecycle, state, iteration budget, and delegation. | Iteration state and final decision. |
| Planner | Maps the question to required facts and candidate corpora using corpus descriptions. | Retrieval plan. |
| Query Rewriter | Converts plan items and sufficiency feedback into precise search queries. | Subqueries with corpus routes. |
| Search Fanout / RAG Agent | Executes searches across selected corpora. | Ranked snippets with provenance. |
| Sufficient Context Judge | Checks whether snippets plus draft answer fully support the requested answer. | Status, covered facts, missing facts, feedback queries. |
| Synthesis Agent | Produces the final answer only after sufficiency is met, or returns partial/unknown with diagnostics. | Grounded answer with citations. |

## Iterative loop

```text
question
  -> planner(question, corpus_catalog)
  -> query_rewriter(plan, prior_feedback=None)
  -> retriever(subqueries)
  -> synthesizer.draft(question, snippets)
  -> sufficiency_judge(question, plan, snippets, draft)
       if sufficient: synthesizer.final(question, snippets, assessment)
       if insufficient and budget remains:
          query_rewriter(plan, prior_feedback=assessment.feedback_queries)
          retriever(...)
          repeat
       else:
          final partial answer or unanswerable diagnostic
```

## Context sufficiency criteria

A context set is sufficient only when all conditions hold:

1. Every required fact in the original question is either directly supported by snippets or explicitly unanswerable from the available corpora.
2. The draft answer contains no unsupported factual claim.
3. Necessary multi-hop bridge facts are present. Example: entity -> id -> record -> attribute.
4. The evidence is relevant to the time, entity, scope, and exclusions in the question.
5. The final answer can cite snippet ids for each material claim.

## Missing-pieces analysis

When context is insufficient, return specific feedback:

- `missing_facts`: exact facts requested but not supported.
- `why_missing`: why current snippets do not cover the fact.
- `feedback_queries`: targeted next searches.
- `candidate_corpora`: where to search next and why.
- `stop_if_not_found`: when to stop searching.

Example:

```json
{
  "status": "insufficient",
  "missing_facts": ["allergic reactions during the hospital stay"],
  "feedback_queries": [
    {
      "query": "John Doe knee surgery adverse reactions rash allergy inpatient stay",
      "target_corpora": ["clinical-notes"],
      "reason": "Medication and diet facts are covered, allergy/adverse-event evidence is missing."
    }
  ]
}
```

## Cross-corpus routing

Each corpus must have a high-quality description. The planner uses descriptions to avoid blind fanout.

Good corpus description:

```text
Quarterly SEC filing PDFs for Alphabet, including income statements, segment revenue, risk factors, and management discussion for fiscal years 2020-2025.
```

Poor corpus description:

```text
Documents.
```

Routing output should be explicit:

```json
{
  "fact": "Project X budget approval date",
  "candidate_corpora": ["finance-approvals", "project-management-logs"],
  "reason": "Budget approvals are likely in finance; timeline references may be in project logs."
}
```

## Synthesis rules

- Use retrieved snippets as the only factual source.
- Add citation markers using snippet ids, for example `[S12]`.
- If evidence conflicts, describe the conflict and cite both sides.
- If max iterations end without sufficiency, return a partial answer plus missing-fact diagnostics.

## Safety and reliability guardrails

- Limit iterations, total tokens, and corpus calls.
- Avoid exposing sensitive snippets unless the caller is authorized.
- Keep the original question immutable; store rewrites separately.
- Preserve audit logs for plan, routing, snippets, sufficiency decisions, and final citations.
- Separate answer generation from sufficiency judging to reduce self-confirmation.
