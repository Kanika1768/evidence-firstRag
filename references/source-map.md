# Source Map

Use these public sources to verify facts before completing the implementation. This file maps research and product sources to the behavior implemented in this repository.

## Implementation alignment

The current scaffold implements the portable subset of the public Agentic RAG pattern:

- Corpus descriptions are treated as routing metadata.
- Retrieval plans decompose a question into required facts and routes.
- Subqueries preserve fact and iteration lineage.
- Snippets preserve corpus id, document id, score, metadata, and query lineage.
- Sufficiency assessment returns status, `sufficiency_score`, covered facts, missing facts, unsupported claims, conflict evidence, and feedback queries.
- The deterministic sufficiency judge classifies sufficient, useful-but-incomplete, insufficient, conflicting, and unanswerable contexts for offline tests and adapter baselines.
- Selective generation abstention maps answerability labels to answered, partial, or unanswerable final outputs before citation validation.
- FRAMES-style evaluation reports fact coverage, fetch coverage, reasoning correctness, citation completeness, and iteration count for a run result.
- Conflict evidence contracts preserve incompatible snippet groups for a required fact through context assessment, structured-output conversion, and final answer payloads.
- Conflict-aware synthesis returns partial answers with citations for each incompatible evidence group instead of merging contradictory snippets into one answer.
- The orchestrator stops on sufficient, irrelevant, unanswerable, max iteration, or no-subquery states.
- Final answers are downgraded when sufficiency fails or citations reference snippets that were not retrieved.

## Registered implementation TODOs

These TODOs map the referenced papers and public product docs to concrete implementation work:

| TODO | Source basis | Implementation target |
|---|---|---|
| `RDD-T-00000007` | Google Agentic RAG roles and structured prompt schemas | Structured-output LLM adapter contracts, schema validation, and one repair attempt for malformed JSON. Completed. |
| `RDD-T-00000008` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Provenance-preserving retriever adapter baseline for non-parametric retrieved memory. Completed. |
| `RDD-T-00000009` | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems | Autorater-style sufficiency classification and selective abstention policy. Completed. |
| `RDD-T-00000010` | Fact, Fetch, and Reason / FRAMES | Multi-hop evaluation harness with fact, fetch, reasoning, citation, and iteration metrics. Completed. |
| `RDD-T-00000011` | Agentic RAG grounded synthesis requirements | Conflict-aware synthesis that cites incompatible evidence instead of merging it silently. Completed. |
| `RDD-T-00000012` | Google Cloud Cross Corpus Retrieval docs | Native Google adapter scaffold with region, IAM, and corpus-resource validation. |

## Decomposed implementation TODOs

These child TODOs preserve the parent roadmap while making each paper-aligned milestone directly implementable and testable.

| Parent TODO | Child TODO | Source basis | Implementation target |
|---|---|---|---|
| `RDD-T-00000007` | `RDD-T-00000014` | Google Agentic RAG roles and structured prompt schemas | Schema registry and dataclass conversion helpers for planner, rewriter, judge, and synthesizer outputs. Completed in `src/agentic_rag/adapters/llm.py`. |
| `RDD-T-00000007` | `RDD-T-00000015` | Google Agentic RAG structured output requirement | Strict JSON parser and validation errors for malformed or schema-invalid model output. Completed in `src/agentic_rag/adapters/llm.py`. |
| `RDD-T-00000007` | `RDD-T-00000016` | Production structured-output repair pattern | One-shot repair protocol contract with explicit malformed output, error list, and schema name. Completed in `src/agentic_rag/adapters/llm.py` and `references/prompts-and-schemas.md`. |
| `RDD-T-00000008` | `RDD-T-00000017` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Provenance-preserving lexical retriever adapter with corpus, document, query, score, metadata, span, and fact lineage. Completed in `src/agentic_rag/adapters/retriever.py`. |
| `RDD-T-00000008` | `RDD-T-00000018` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Deterministic scoring, ordering, span extraction, and duplicate-document behavior. Completed in `src/agentic_rag/adapters/retriever.py` and `tests/test_retriever_adapter.py`. |
| `RDD-T-00000009` | `RDD-T-00000019` | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems | Answerability labels for sufficient, useful-but-incomplete, insufficient, conflicting, and unanswerable contexts. Completed in `src/agentic_rag/contracts.py`. |
| `RDD-T-00000009` | `RDD-T-00000020` | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems | Autorater-style sufficiency judge with missing facts, unsupported claims, covered facts, and feedback queries. Completed in `src/agentic_rag/sufficiency.py`. |
| `RDD-T-00000009` | `RDD-T-00000021` | Sufficient Context: A New Lens on Retrieval Augmented Generation Systems | Selective generation abstention policy that maps answerability to answered, partial, or unanswerable results. Completed in `src/agentic_rag/sufficiency.py` and `src/agentic_rag/orchestrator.py`. |
| `RDD-T-00000010` | `RDD-T-00000022` | Fact, Fetch, and Reason / FRAMES | Fixture format and metrics for fact coverage, fetch coverage, reasoning correctness, citation completeness, and iteration count. Completed in `src/agentic_rag/evaluation.py`. |
| `RDD-T-00000010` | `RDD-T-00000023` | Fact, Fetch, and Reason / FRAMES | Iterative-vs-single-shot multi-hop evaluation proving the value of follow-up retrieval. Completed in `tests/test_evaluation.py`. |
| `RDD-T-00000011` | `RDD-T-00000024` | Agentic RAG grounded synthesis requirements | Conflict evidence contracts that keep incompatible snippet groups visible and citable. Completed in `src/agentic_rag/contracts.py` and `src/agentic_rag/adapters/llm.py`. |
| `RDD-T-00000011` | `RDD-T-00000025` | Agentic RAG grounded synthesis requirements | Conflict-aware judge and synthesis behavior for contradictory evidence. Completed in `src/agentic_rag/sufficiency.py`, `src/agentic_rag/adapters/in_memory.py`, and `tests/test_orchestrator.py`. |
| `RDD-T-00000012` | `RDD-T-00000026` | Google Cloud Cross Corpus Retrieval docs | Native mode configuration validation for project, location, corpus resources, request mode, and service-account assumptions. |
| `RDD-T-00000012` | `RDD-T-00000027` | Google Cloud Cross Corpus Retrieval docs | Cross Corpus request adapter seam for `AsyncRetrieveContexts`, `AskContexts`, or equivalent payloads with injected clients. |
| `RDD-T-00000012` | `RDD-T-00000028` | Google Cloud Cross Corpus Retrieval docs | Public documentation for portable default mode, optional native mode, required configuration, and no-SDK default imports. |

## Google Research announcement

URL: https://research.google/blog/unlocking-dependable-responses-with-gemini-enterprise-agent-platforms-agentic-rag/

Key public facts to preserve:

- Published 2026-06-05.
- Describes a Google Research + Google Cloud Agentic RAG framework for Gemini Enterprise Agent Platform.
- Publicly describes a multi-agent workflow that breaks down complex enterprise queries and iteratively searches for sufficient context.
- Publicly listed roles include Orchestrator, Planner Agent, Query Rewriter, Search Fanout/RAG Agent, Sufficient Context Agent, and Synthesis/LLM aggregation.
- Core differentiator is persistence through Sufficient Context checking: the system identifies missing pieces and re-searches instead of stopping after first retrieval.
- Reported public results include up to 34% accuracy improvement versus standard RAG on factuality datasets and 90.1% accuracy in the described cross-corpus FramesQA setting.

Implementation relevance:

- Motivates the multi-agent loop: orchestrator, planner, query rewriter, search fanout, sufficient context agent, and synthesizer.
- Motivates missing-fact feedback and iterative follow-up retrieval.
- Connects Agentic RAG evaluation to FramesQA/FRAMES and cross-corpus routing.
- Tracked by `RDD-T-00000007`, `RDD-T-00000009`, `RDD-T-00000010`, `RDD-T-00000011`, and `RDD-T-00000012`.

## Google Cloud Cross Corpus Retrieval docs

URL: https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/cross-corpus-retrieval

Key public facts to preserve:

- Feature is RAG Engine Cross Corpus Retrieval.
- Docs describe `AsyncRetrieveContexts` and `AskContexts` APIs.
- Docs state the system uses an agentic approach across multiple corpora.
- Architecture components include Orchestrator/Router, Planning Agent, Retrieval Engine, Reasoning Agent, and LLM Generator.
- Corpus `description` is important for corpus selection.
- Public docs state the feature is available only in `us-central1` at the time of the referenced page.
- Requires granting Vertex AI User role to the RAG Engine service account.

Implementation relevance:

- Motivates provider adapters rather than embedding product-specific calls in the orchestrator.
- Motivates high-quality corpus descriptions for cross-corpus routing.
- Remains planned follow-up work for native Google mode.
- Tracked by `RDD-T-00000012`.

## Referenced papers

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

URL: https://arxiv.org/abs/2005.11401

Key public facts to preserve:

- Introduces the RAG framing that combines parametric generation with non-parametric retrieved memory.
- Establishes the retrieve-then-generate baseline that this project extends with planning, sufficiency checks, iteration, and claim-level provenance.

Implementation relevance:

- Provides the baseline terminology for RAG and retrieved evidence.
- Motivates explicit provenance and updateable external knowledge.
- Tracked by `RDD-T-00000008`.

### Sufficient Context: A New Lens on Retrieval Augmented Generation Systems

URL: https://arxiv.org/abs/2411.06037

Key public facts to preserve:

- Defines sufficient context as context that contains enough information to provide a definitive answer.
- Studies a sufficient-context autorater and shows that relevance alone is not enough to decide answerability.
- Motivates using sufficiency signals to reduce hallucinations and guide abstention.

Implementation relevance:

- Directly motivates `ContextAssessment`, `sufficiency_score`, `missing_facts`, `unsupported_claims`, answerability labels, targeted feedback queries, selective abstention, and partial/unanswerable answers.
- Supports the guardrail that a final answer should not be marked answered when context is insufficient.
- Tracked by `RDD-T-00000009`.

### Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation

URL: https://arxiv.org/abs/2409.12941

Key public facts to preserve:

- Introduces FRAMES, a multi-hop RAG evaluation benchmark for factuality, retrieval, and reasoning.
- Evaluates questions that require retrieving and combining facts from multiple sources.

Implementation relevance:

- Motivates evaluation fixtures for multi-hop and cross-corpus RAG.
- Directly motivates the `EvaluationFixture`, `ExpectedFetch`, `EvaluationMetrics`, and `evaluate_run` helpers in `src/agentic_rag/evaluation.py`.
- Supports comparing single-shot retrieval with iterative follow-up retrieval through `compare_runs`.
- Provides the research basis for FramesQA-style tests mentioned in the Google Research Agentic RAG announcement.
- Tracked by `RDD-T-00000010`.

## Agent Skills specification

URL: https://agentskills.io/specification

Key public facts to preserve:

- A skill is a folder containing at minimum `SKILL.md`.
- `SKILL.md` requires YAML frontmatter with `name` and `description`.
- Optional directories include `scripts/`, `references/`, and `assets/`.
- Skills should use progressive disclosure: keep `SKILL.md` focused and load reference files on demand.

Implementation relevance:

- Motivates the repository layout with `SKILL.md`, `references/`, `agents/`, `src/`, and tests.

## Codex Skills docs

URL: https://developers.openai.com/codex/skills

Key public facts to preserve:

- Codex supports Agent Skills with `SKILL.md`, optional `scripts/`, `references/`, `assets/`, and `agents/openai.yaml`.
- Codex scans `.agents/skills` in the current directory and parent directories up to the repo root for repo-scoped skills.
- Codex can invoke skills explicitly or implicitly based on the description.

Implementation relevance:

- Motivates keeping `SKILL.md` concise and placing detailed guidance in `references/`.
- Motivates `agents/openai.yaml` for Codex-facing skill metadata.
