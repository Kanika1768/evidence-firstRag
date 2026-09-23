# Repo Agent Instructions

Use the `agentic-rag` skill whenever the task mentions Agentic RAG, Google Agentic RAG, Sufficient Context Agent, Cross-Corpus Retrieval, multi-hop RAG, multi-source RAG, RAG planning, query rewriting, context sufficiency, grounded synthesis, or Agent Skills for RAG.

## Development Rules

- Keep this repo as the canonical `agentic-rag` skill and Python scaffold. Do not split skill packaging into a separate `Agentic-RAG-Skill` repo.
- Do not claim to reproduce Google's internal implementation. Implement the public pattern described in the linked Google Research and Google Cloud documentation.
- Preserve the public contracts in `src/agentic_rag/contracts.py` unless a test or requirement explicitly justifies a change.
- Add provider-specific behavior under `src/agentic_rag/adapters/` instead of hard-coding a vendor into the orchestrator.
- Keep `SKILL.md` concise and route detailed guidance through `references/`.
- Run `python scripts/validate_skill.py` and `python -m unittest discover -s tests -v` before publishing.

## Completion Target

When asked to complete this scaffold, implement missing capabilities in this order:

1. LLM client abstraction and structured-output parsing.
2. Retriever adapter for the target corpus store.
3. Sufficient Context Judge with strict missing-fact feedback.
4. Integration tests using at least two corpora with distracting data.
5. Optional Google Vertex RAG native adapter.
