# Agentic RAG Implementation Summary

This file summarizes the public Agentic RAG pattern in implementation terms. It is not a reproduction of any internal Google system.

## Core Difference

Standard RAG often follows a single-pass flow:

```text
question -> retrieve once -> generate answer
```

Agentic RAG adds planning, targeted corpus routing, sufficiency checks, and follow-up retrieval:

```text
question
  -> plan required facts
  -> rewrite targeted retrieval queries
  -> retrieve from selected corpora
  -> draft only from retrieved snippets
  -> judge context sufficiency
       sufficient: synthesize final grounded answer
       insufficient: generate missing-fact queries and retrieve again
       exhausted: return partial or unanswerable diagnostics
```

## Required Components

1. **Orchestrator / Router**
   - Owns request state, iteration count, cost limits, and component delegation.

2. **Planner Agent**
   - Breaks the original question into required facts.
   - Routes each fact to candidate corpora by reading corpus descriptions.

3. **Query Rewriter**
   - Converts plan items and sufficiency feedback into precise search queries.
   - Preserves lineage from original question to plan item to subquery.

4. **Search Fanout / RAG Agent**
   - Searches only the selected corpora unless the planner justifies broad fanout.
   - Preserves snippet id, corpus id, document id, score, span, and metadata.

5. **Sufficient Context Judge**
   - Checks whether every required fact is supported.
   - Flags unsupported draft claims.
   - Emits missing facts and targeted feedback queries when context is insufficient.

6. **Synthesis Agent**
   - Produces the final answer only from supported snippets.
   - Cites snippet ids for each material claim.

## Non-Negotiable Behaviors

- Do not treat high similarity as sufficient context by itself.
- Do not search every corpus for every question without a route plan.
- Do not finalize unsupported claims.
- Do not return "not found" before targeted follow-up retrieval has been attempted.
- Preserve claim-level provenance from snippet to final answer.
