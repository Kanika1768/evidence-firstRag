# Prompts and JSON Schemas

Use these as structured-output contracts. Convert them to provider-specific JSON schema, function calling, tool calling, or Pydantic models as needed.

## Planner prompt

```text
You are the Planner Agent for an Agentic RAG system.

Input:
- Original user question
- Corpus catalog with id, description, metadata
- Optional prior insufficiency feedback

Task:
1. Decompose the question into required facts.
2. Identify candidate corpora for each fact using corpus descriptions.
3. Explain why each corpus is relevant.
4. Define expected evidence and stop conditions.
5. Do not route to every corpus unless justified.

Return only JSON matching RetrievalPlan.
```

### RetrievalPlan schema

```json
{
  "type": "object",
  "required": ["question", "required_facts", "routes", "stop_conditions"],
  "properties": {
    "question": {"type": "string"},
    "required_facts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "description", "priority"],
        "properties": {
          "id": {"type": "string"},
          "description": {"type": "string"},
          "priority": {"type": "string", "enum": ["must", "should", "nice"]}
        }
      }
    },
    "routes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fact_id", "candidate_corpus_ids", "reason"],
        "properties": {
          "fact_id": {"type": "string"},
          "candidate_corpus_ids": {"type": "array", "items": {"type": "string"}},
          "reason": {"type": "string"}
        }
      }
    },
    "stop_conditions": {"type": "array", "items": {"type": "string"}}
  }
}
```

## Query Rewriter prompt

```text
You are the Query Rewriter for an Agentic RAG system.

Create precise retrieval queries for each plan route. Use prior sufficiency feedback to target missing facts. Preserve fact_id and corpus route. Avoid broad queries unless the plan requires discovery.

Return only JSON matching QueryRewriteResult.
```

### QueryRewriteResult schema

```json
{
  "type": "object",
  "required": ["subqueries"],
  "properties": {
    "subqueries": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "fact_id", "query", "target_corpus_ids", "reason"],
        "properties": {
          "id": {"type": "string"},
          "fact_id": {"type": "string"},
          "query": {"type": "string"},
          "target_corpus_ids": {"type": "array", "items": {"type": "string"}},
          "reason": {"type": "string"}
        }
      }
    }
  }
}
```

## Sufficient Context Judge prompt

```text
You are the Sufficient Context Judge.

Input:
- Original question
- Retrieval plan
- Retrieved snippets with ids and corpus ids
- Intermediate draft answer

Task:
1. Check every required fact.
2. Mark which facts are supported, missing, conflicting, or out of scope.
3. Identify unsupported claims in the draft.
4. For conflicting facts, cite each incompatible snippet group separately.
5. Decide status: sufficient, insufficient, irrelevant, or unanswerable.
6. If insufficient, generate targeted feedback queries and candidate corpora.
7. Never mark sufficient if any must-have fact is unsupported.

Return only JSON matching ContextAssessment.
```

### ContextAssessment schema

```json
{
  "type": "object",
  "required": ["status", "sufficiency_score", "covered_facts", "missing_facts", "unsupported_claims", "feedback_queries", "reason"],
  "properties": {
    "status": {"type": "string", "enum": ["sufficient", "insufficient", "irrelevant", "unanswerable"]},
    "answerability": {"type": "string", "enum": ["sufficient", "useful_but_incomplete", "insufficient", "conflicting", "unanswerable"]},
    "sufficiency_score": {"type": "number", "minimum": 0, "maximum": 1},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Backward-compatible alias for sufficiency_score."},
    "covered_facts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fact_id", "snippet_ids"],
        "properties": {
          "fact_id": {"type": "string"},
          "snippet_ids": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "missing_facts": {"type": "array", "items": {"type": "string"}},
    "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    "conflicts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fact_id", "groups"],
        "properties": {
          "fact_id": {"type": "string"},
          "reason": {"type": "string"},
          "groups": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["label", "snippet_ids"],
              "properties": {
                "label": {"type": "string"},
                "value": {"type": "string"},
                "snippet_ids": {"type": "array", "items": {"type": "string"}}
              }
            }
          }
        }
      }
    },
    "feedback_queries": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["query", "target_corpus_ids", "reason"],
        "properties": {
          "query": {"type": "string"},
          "target_corpus_ids": {"type": "array", "items": {"type": "string"}},
          "reason": {"type": "string"}
        }
      }
    },
    "reason": {"type": "string"}
  }
}
```

## Synthesis prompt

```text
You are the Synthesis Agent for a grounded Agentic RAG system.

Answer the original question using only supported retrieved snippets. Cite snippet ids for every material claim. Apply the answerability label before final output: sufficient can be answered, useful-but-incomplete and conflicting contexts must be partial with diagnostics, and insufficient or unanswerable contexts must be unanswerable instead of guessed.
When context is conflicting, preserve each incompatible evidence group and cite the snippets for both sides.

Return only JSON matching GroundedAnswer.
```

### GroundedAnswer schema

```json
{
  "type": "object",
  "required": ["answer", "citations", "status", "missing_facts", "sufficiency_score"],
  "properties": {
    "answer": {"type": "string"},
    "citations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["claim", "snippet_ids"],
        "properties": {
          "claim": {"type": "string"},
          "snippet_ids": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "status": {"type": "string", "enum": ["answered", "partial", "unanswerable"]},
    "missing_facts": {"type": "array", "items": {"type": "string"}},
    "sufficiency_score": {"type": "number", "minimum": 0, "maximum": 1},
    "conflicts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["fact_id", "groups"],
        "properties": {
          "fact_id": {"type": "string"},
          "reason": {"type": "string"},
          "groups": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["label", "snippet_ids"],
              "properties": {
                "label": {"type": "string"},
                "value": {"type": "string"},
                "snippet_ids": {"type": "array", "items": {"type": "string"}}
              }
            }
          }
        }
      }
    }
  }
}
```

## Structured output repair protocol

Provider adapters may retry malformed structured output at most once. The repair request must include:

- `schema_name`: the requested schema name, such as `RetrievalPlan`, `QueryRewriteResult`, `ContextAssessment`, or `GroundedAnswer`.
- `errors`: validation error messages from the first parse attempt.
- `malformed_output`: the exact model output that failed parsing or validation.

The repair prompt contract is:

```text
Repair the structured JSON output so it matches the requested schema.
Return only corrected JSON. Do not include markdown fences or explanations.
Schema name: <schema_name>
Validation errors:
- <error>
Malformed output:
<malformed_output>
```

The repaired output is parsed through the same strict schema validator. If the repaired output is still invalid, the adapter must surface that second validation error and must not attempt another repair.
