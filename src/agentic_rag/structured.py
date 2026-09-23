"""Structured-output parser helpers for provider adapters.

This module keeps backward-compatible parsing behavior for the
`google-agentic-rag-skill` public surface while mapping parsed values into the
newer contract dataclasses shipped by this repository.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence as _Sequence, Protocol

from .contracts import (
    ClaimCitation,
    Claim,
    ContextAssessment,
    CoveredFact,
    DraftAnswer,
    FeedbackQuery,
    GroundedAnswer,
    RequiredFact,
    RetrievalPlan,
    RetrievalRoute,
    SubQuery,
)


class StructuredLLM(Protocol):
    """Provider-neutral interface for deterministic JSON outputs."""

    def complete_json(self, *, task: str, prompt: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | str:
        """Return a JSON-compatible object for a named structured task."""


def parse_json_object(text: str) -> Mapping[str, Any]:
    """Parse a JSON object and reject arrays/scalars early."""

    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("structured output must be a JSON object")
    return value


def plan_from_mapping(data: Mapping[str, Any]) -> RetrievalPlan:
    if isinstance(data, str):
        data = parse_json_object(data)
    return RetrievalPlan(
        question=_string(data.get("question"), "question"),
        required_facts=tuple(
            RequiredFact(
                id=_string(item.get("id"), "required_facts.id"),
                description=_string(item.get("description"), "required_facts.description"),
                priority=item.get("priority", "must"),
            )
            for item in _items(data.get("required_facts"), "required_facts")
        ),
        routes=tuple(
            RetrievalRoute(
                fact_id=_string(item.get("fact_id"), "routes.fact_id"),
                candidate_corpus_ids=_string_tuple(item.get("candidate_corpus_ids"), "routes.candidate_corpus_ids"),
                reason=_string(item.get("reason"), "routes.reason"),
            )
            for item in _items(data.get("routes"), "routes")
        ),
        stop_conditions=_string_tuple(data.get("stop_conditions", ()), "stop_conditions"),
    )


def subqueries_from_mapping(data: Mapping[str, Any]) -> tuple[SubQuery, ...]:
    if isinstance(data, str):
        data = parse_json_object(data)
    return tuple(
        SubQuery(
            id=_string(item.get("id"), "subqueries.id"),
            fact_id=_string(item.get("fact_id"), "subqueries.fact_id"),
            query=_string(item.get("query"), "subqueries.query"),
            target_corpus_ids=_string_tuple(item.get("target_corpus_ids"), "subqueries.target_corpus_ids"),
            reason=_string(item.get("reason"), "subqueries.reason"),
        )
        for item in _items(data.get("subqueries"), "subqueries")
    )


def assessment_from_mapping(data: Mapping[str, Any]) -> ContextAssessment:
    if isinstance(data, str):
        data = parse_json_object(data)
    if "sufficiency_score" in data:
        sufficiency_score = _number(data.get("sufficiency_score"), "sufficiency_score")
    elif "confidence" in data:
        sufficiency_score = _number(data.get("confidence"), "confidence")
    else:
        raise ValueError("ContextAssessment must include sufficiency_score or confidence")

    return ContextAssessment(
        status=_status(data.get("status")),
        sufficiency_score=sufficiency_score,
        covered_facts=tuple(
            CoveredFact(
                fact_id=_string(item.get("fact_id"), "covered_facts.fact_id"),
                snippet_ids=_string_tuple(item.get("snippet_ids"), "covered_facts.snippet_ids"),
            )
            for item in _items(data.get("covered_facts", []), "covered_facts")
        ),
        missing_facts=_string_tuple(data.get("missing_facts", ()), "missing_facts"),
        unsupported_claims=_string_tuple(data.get("unsupported_claims", ()), "unsupported_claims"),
        feedback_queries=tuple(
            FeedbackQuery(
                query=_string(item.get("query"), "feedback_queries.query"),
                target_corpus_ids=_string_tuple(item.get("target_corpus_ids"), "feedback_queries.target_corpus_ids"),
                reason=_string(item.get("reason"), "feedback_queries.reason"),
                fact_id=_string(item.get("fact_id"), "feedback_queries.fact_id") if item.get("fact_id") is not None else None,
            )
            for item in _items(data.get("feedback_queries", []), "feedback_queries")
        ),
        reason=str(data.get("reason", "")),
        answerability=str(data.get("answerability")) if data.get("answerability") is not None else None,
    )


def draft_from_mapping(data: Mapping[str, Any]) -> DraftAnswer:
    if isinstance(data, str):
        data = parse_json_object(data)
    claims = _claims_from_mapping(data)
    return DraftAnswer(
        text=_string(data.get("text"), "text"),
        claims=claims,
        cited_snippet_ids=_string_tuple(data.get("cited_snippet_ids", ()), "cited_snippet_ids"),
        unsupported_claims=_string_tuple(data.get("unsupported_claims", ()), "unsupported_claims"),
    )


def answer_from_mapping(data: Mapping[str, Any], *, iterations: int = 0) -> GroundedAnswer:
    if isinstance(data, str):
        data = parse_json_object(data)
    if "sufficiency_score" in data:
        sufficiency_score = _number(data.get("sufficiency_score"), "sufficiency_score")
    elif "confidence" in data:
        sufficiency_score = _number(data.get("confidence"), "confidence")
    else:
        sufficiency_score = 0.0

    status = _answer_status(data.get("status"))
    return GroundedAnswer(
        answer=_string(data.get("answer"), "answer"),
        status=_answer_status(status),
        citations=tuple(
            _claim_from_mapping(item, "citations")
            for item in _items(data.get("citations", []), "citations")
        ),
        missing_facts=_string_tuple(data.get("missing_facts", ()), "missing_facts"),
        sufficiency_score=sufficiency_score,
        iterations=iterations,
    )


def _claim_from_mapping(item: Mapping[str, Any], field: str) -> ClaimCitation:
    return ClaimCitation(
        claim=_string(item.get("claim"), f"{field}.claim"),
        snippet_ids=_string_tuple(item.get("snippet_ids"), f"{field}.snippet_ids"),
    )


def _status(value: object) -> str:
    text = _string(value, "status") if value is not None else ""
    value_text = text.lower()
    if value_text not in {"sufficient", "insufficient", "irrelevant", "unanswerable"}:
        raise ValueError(f"invalid status: {text}")
    return value_text


def _answer_status(value: object) -> str:
    text = _string(value, "status") if value is not None else ""
    value_text = text.lower()
    if value_text not in {"answered", "partial", "unanswerable"}:
        raise ValueError(f"invalid answer status: {text}")
    return value_text


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of strings")
    result = tuple(_string(item, field) for item in value)
    return result


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _items(value: Any, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, _Sequence) or isinstance(value, str):
        raise ValueError(f"{field} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field} items must be objects")
    return value


def _claims_from_mapping(raw: Mapping[str, Any]) -> tuple[Claim, ...]:
    claims_value = raw.get("claims")
    if claims_value is None:
        return tuple(
            Claim(_string(text, "text"), _string_tuple(raw.get("cited_snippet_ids", ()), "cited_snippet_ids"))
            for text in [_string(raw.get("text"), "text")]
            if raw.get("text")
        )
    if not isinstance(claims_value, _Sequence) or isinstance(claims_value, str):
        raise ValueError("claims must be a list of claim objects")
    if not all(isinstance(claim, Mapping) for claim in claims_value):
        raise ValueError("claims[] must be an object")
    parsed: list[Claim] = []
    for claim in claims_value:
        parsed.append(
            Claim(
                text=_string(claim.get("text"), "claims.text"),
                snippet_ids=_string_tuple(claim.get("snippet_ids", ()), "claims.snippet_ids"),
            )
        )
    return tuple(parsed)
