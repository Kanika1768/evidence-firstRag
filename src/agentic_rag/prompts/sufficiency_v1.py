"""EvidenceFirst Sufficiency Evaluation Prompt Template (v1.0).

This module defines the versioned system prompt, user prompt builder, and
JSON schema for evaluating context sufficiency at the retrieval-generation boundary
under the Diligent Reader standard (Joren et al., ICLR 2025).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

SUFFICIENCY_PROMPT_VERSION = "v1.0"

SUFFICIENCY_SYSTEM_PROMPT = """You are the EvidenceFirst Sufficiency Judge, an expert autorater evaluating retrieved context for Retrieval-Augmented Generation.

Your task is to determine whether the provided context is strictly SUFFICIENT for a diligent reader to construct a definitive answer to the user's question.

CRITICAL DILIGENT READER CRITERION:
"A context is sufficient if a diligent reader can construct a definitive answer using ONLY the provided context without requiring outside knowledge."

EVALUATION RULES:
1. STRICT SUFFICIENCY (No Assumptions):
   - Every fact, entity, relation, condition, or intermediate step necessary to answer the question must be explicitly stated in the context.
   - Do NOT use outside world knowledge or parametric memory to bridge unstated gaps.
   - If the question contains a false premise, the context is sufficient ONLY if it explicitly refutes or clarifies the premise.
2. MULTI-HOP DEDUCTION:
   - Deductions connecting facts explicitly stated in the context are permitted.
   - Any inferential leap that requires an unstated bridging fact renders the context INSUFFICIENT.
3. MISSING INFORMATION IDENTIFICATION:
   - If the context is INSUFFICIENT, you must explicitly enumerate the precise missing facts or queries needed to bridge the evidential gap.
   - If SUFFICIENT, the missing_information list must be empty.
4. PROBABILITY / CONFIDENCE SCORE:
   - Provide a calibrated float probability in [0.0, 1.0] representing your confidence in context sufficiency (1.0 = fully sufficient, 0.0 = completely insufficient).

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this exact schema:
{
  "sufficient": <true or false>,
  "probability": <float between 0.0 and 1.0>,
  "missing_information": [<string>, ...],
  "rationale": "<concise explanation referencing specific evidence or missing facts>"
}
"""

SUFFICIENCY_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "StructuredSufficiencyResult",
    "description": "Structured output from the EvidenceFirst Sufficiency Judge.",
    "type": "object",
    "properties": {
        "sufficient": {
            "type": "boolean",
            "description": "True if context contains all facts required to answer the question; False otherwise.",
        },
        "probability": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score in context sufficiency.",
        },
        "missing_information": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific missing facts or claims required to answer the question.",
        },
        "rationale": {
            "type": "string",
            "description": "Concise justification for the sufficiency judgment.",
        },
    },
    "required": ["sufficient", "probability", "missing_information", "rationale"],
    "additionalProperties": False,
}


def get_sufficiency_system_prompt() -> str:
    """Return the system prompt for sufficiency evaluation."""
    return SUFFICIENCY_SYSTEM_PROMPT.strip()


def get_sufficiency_json_schema() -> dict[str, Any]:
    """Return the JSON schema defining the required output structure."""
    return dict(SUFFICIENCY_JSON_SCHEMA)


def build_sufficiency_user_prompt(
    question: str,
    context: str,
    required_facts: Sequence[str] | None = None,
) -> str:
    """Construct the user evaluation prompt for a specific question and context snippet."""
    prompt_lines = [
        f"QUESTION: {question.strip()}",
        "",
        "RETRIEVED CONTEXT:",
        context.strip(),
        "",
    ]

    if required_facts:
        prompt_lines.append("PLANNED REQUIRED FACTS:")
        for idx, fact in enumerate(required_facts, start=1):
            prompt_lines.append(f"  {idx}. {fact}")
        prompt_lines.append("")

    prompt_lines.append(
        "Evaluate whether the retrieved context is SUFFICIENT for a diligent reader to construct "
        "a definitive answer to the question using ONLY the provided text.\n"
        "Provide your evaluation as a valid JSON object."
    )
    return "\n".join(prompt_lines)


def build_sufficiency_prompt(
    question: str,
    context: str,
    required_facts: Sequence[str] | None = None,
) -> str:
    """Construct a full composite prompt containing both system instructions and user input."""
    system_part = get_sufficiency_system_prompt()
    user_part = build_sufficiency_user_prompt(question, context, required_facts)
    return f"{system_part}\n\n{'=' * 40}\n\n{user_part}"
