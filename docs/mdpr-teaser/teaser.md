# Agentic RAG README Teaser

## Iterative retrieval is tested against one-shot retrieval

Iterative retrieval improves fetch coverage and citation completeness from 0.5 to 1.0 in the deterministic evaluation fixture, while keeping missing context explicit through sufficiency checks.

| Signal | Value |
|---|---|
| One-shot fetch | 0.5 |
| Iterative fetch | 1.0 |
| Citation lift | 0.5 -> 1.0 |
| Pipeline | plan -> route -> rewrite -> retrieve -> judge -> iterate -> synthesize |
