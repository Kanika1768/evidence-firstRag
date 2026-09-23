---
name: agentic-rag
description: >-
  Build, review, or refactor Agentic RAG systems with planning, query rewriting,
  cross-corpus routing, retrieval fanout, Sufficient Context checks, iterative
  follow-up retrieval, and grounded synthesis with citations. Use for multi-hop
  RAG, multi-source RAG, context sufficiency, or Agent Skill scaffolds based on
  the public Google Research and Google Cloud pattern.
---

# Agentic RAG

Use this skill to implement or refactor a RAG system into the public Agentic RAG pattern described by Google Research and Google Cloud: plan, route, retrieve, check sufficiency, iterate, and synthesize grounded answers.

## Read first

- For the detailed behavior model, read `references/agentic-rag-behavior.md`.
- For a concise implementation summary, read `references/agentic-rag-behavior-summary.md`.
- For JSON prompts and output schemas, read `references/prompts-and-schemas.md`.
- For completion tasks and TODO order, read `references/codex-completion-brief.md`.
- For source URLs and fact map, read `references/source-map.md`.

## Activation signals

Activate this skill when the task includes any of these terms or intents:

- Agentic RAG, Gemini Enterprise Agent Platform RAG, Cross-Corpus Retrieval, Agentic Retrieval.
- Sufficient Context Agent, Sufficient Context Awareness, context sufficiency, iterative retrieval.
- Multi-hop RAG, multi-source RAG, cross-corpus RAG, query planning, query rewriting, search fanout.
- Build a Codex/Claude/Gemini Agent Skill for RAG.
- Refactor a "vanilla RAG" pipeline that fails when information is split across corpora.

## Core workflow

Follow this workflow exactly unless the user asks for a narrower task.

1. **Classify mode**
   - Native Google mode: use Gemini Enterprise Agent Platform RAG Engine Cross Corpus Retrieval APIs if the user has Google Cloud project, location, RAG corpora, IAM, and region requirements.
   - Portable mode: implement the public pattern using local or third-party retrievers.

2. **Build corpus catalog**
   - Require a concise `description` for each corpus.
   - Treat descriptions as routing metadata.
   - Do not naively search every corpus unless the query is broad or the planner justifies it.

3. **Plan**
   - Decompose the user question into required facts.
   - Map each fact to candidate corpora.
   - Produce a retrieval plan with expected evidence and stop conditions.

4. **Rewrite and fan out**
   - Generate targeted search queries for each required fact and corpus route.
   - Include follow-up queries when a previous sufficiency check reports missing facts.
   - Preserve query lineage: original question -> plan item -> rewritten query -> retrieved snippets.

5. **Retrieve**
   - Retrieve snippets from selected corpora.
   - Keep snippet ids, corpus ids, document ids, scores, text spans, and metadata.
   - Deduplicate near-identical snippets before synthesis.

6. **Draft**
   - Create an intermediate answer only from retrieved snippets.
   - Mark unsupported claims as missing rather than filling gaps.

7. **Sufficient Context check**
   - Judge the original question, retrieval plan, snippets, and draft together.
   - Return one of: `sufficient`, `insufficient`, `irrelevant`, or `unanswerable`.
   - If insufficient, list missing facts and concrete feedback queries.
   - If a corpus is irrelevant, state why and suggest a better route when possible.

8. **Iterate**
   - If status is insufficient and iteration budget remains, use the feedback to re-plan/rewrite/retrieve.
   - Stop when sufficient, unanswerable, or max iterations is reached.
   - Keep an audit trail of each iteration.

9. **Synthesize final answer**
   - Answer only with supported facts.
   - Attach snippet citations or source identifiers to factual claims.
   - If the answer is partial, say exactly what is missing and which follow-up retrieval would be needed.

## Implementation rules

- Start from `src/agentic_rag/contracts.py` and `src/agentic_rag/orchestrator.py` when this scaffold is present.
- Use `src/agentic_rag/adapters/in_memory.py` for deterministic offline tests, examples, and provider-adapter contract checks.
- Implement provider integrations as adapters, not inside the orchestrator.
- Use deterministic structured JSON for planner, query rewriter, sufficiency judge, and synthesis outputs.
- Never use a final answer from a failed sufficiency check as if it were grounded.
- Enforce max iteration and max cost limits.
- Log all plan items, subqueries, hits, missing facts, and final citations.

## Portable Python scaffold

This skill includes a dependency-free Python scaffold for productizing the loop:

- `contracts.py` defines corpus catalogs, retrieval plans, subqueries, snippets, claim citations, `sufficiency_score`, feedback queries, iteration traces, and grounded answers.
- `orchestrator.py` preserves query lineage, deduplicates snippets, checks sufficiency every iteration, and prevents an `answered` final result when the assessment is not sufficient.
- `adapters/in_memory.py` provides deterministic planner, query rewriter, retriever, judge, drafter, and synthesizer components for tests and examples.

Run the scaffold checks with:

```text
python -m unittest discover -s tests -v
```

## Anti-patterns

Avoid these failures:

- Single-shot retrieval followed by a confident answer.
- Searching every corpus for every question without a route plan.
- Treating high vector similarity as sufficient context.
- Generating a final answer without checking every requested fact.
- Losing provenance between snippets and final claims.
- Returning "not found" before targeted follow-up queries have been attempted.
