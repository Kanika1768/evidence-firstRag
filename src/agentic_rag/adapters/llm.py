"""Structured-output schema helpers for LLM provider adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Callable, Mapping, Sequence

from agentic_rag.contracts import (
    AnswerabilityLabel,
    ConflictEvidence,
    ConflictingEvidenceGroup,
    ContextAssessment,
    CoveredFact,
    FeedbackQuery,
    GroundedAnswer,
    GroundedCitation,
    QueryRewriteResult,
    RequiredFact,
    RetrievalPlan,
    Route,
    Subquery,
)


class SchemaValidationError(ValueError):
    """Raised when a structured-output mapping cannot satisfy a schema contract."""

    def __init__(self, schema_name: str, message: str) -> None:
        self.schema_name = schema_name
        super().__init__(f"{schema_name}: {message}")


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    target_type: type
    required_fields: Sequence[str]
    from_mapping: Callable[[Mapping[str, Any]], Any]
    to_mapping: Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class StructuredOutputRepairRequest:
    schema_name: str
    malformed_output: str
    errors: Sequence[str]
    prompt: str


def get_schema(schema_name: str) -> SchemaSpec:
    try:
        return SCHEMA_REGISTRY[schema_name]
    except KeyError as exc:
        raise SchemaValidationError(schema_name, "unknown schema") from exc


def parse_structured_output(schema_name: str, output: str) -> Any:
    try:
        raw = json.loads(output)
    except JSONDecodeError as exc:
        raise SchemaValidationError(
            schema_name,
            f"malformed JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
        ) from exc
    if not isinstance(raw, Mapping):
        raise SchemaValidationError(schema_name, "top-level JSON must be an object")
    return to_dataclass(schema_name, raw)


def parse_structured_output_with_repair(
    schema_name: str,
    output: str,
    *,
    repair: Callable[[StructuredOutputRepairRequest], str],
) -> Any:
    try:
        return parse_structured_output(schema_name, output)
    except SchemaValidationError as exc:
        request = build_structured_output_repair_request(schema_name, output, (str(exc),))
        repaired_output = repair(request)
        return parse_structured_output(schema_name, repaired_output)


def build_structured_output_repair_request(
    schema_name: str,
    malformed_output: str,
    errors: Sequence[str],
) -> StructuredOutputRepairRequest:
    return StructuredOutputRepairRequest(
        schema_name=schema_name,
        malformed_output=malformed_output,
        errors=tuple(errors),
        prompt=build_structured_output_repair_prompt(schema_name, malformed_output, errors),
    )


def build_structured_output_repair_prompt(
    schema_name: str,
    malformed_output: str,
    errors: Sequence[str],
) -> str:
    error_lines = "\n".join(f"- {error}" for error in errors) or "- Unknown validation error"
    return "\n".join(
        (
            "Repair the structured JSON output so it matches the requested schema.",
            "Return only corrected JSON. Do not include markdown fences or explanations.",
            f"Schema name: {schema_name}",
            "Validation errors:",
            error_lines,
            "Malformed output:",
            malformed_output,
        )
    )


def to_dataclass(schema_name: str, raw: Mapping[str, Any]) -> Any:
    if not isinstance(raw, Mapping):
        raise SchemaValidationError(schema_name, "schema input must be an object")
    spec = get_schema(schema_name)
    _require_fields(schema_name, raw, spec.required_fields)
    return spec.from_mapping(raw)


def to_mapping(schema_name: str, value: Any) -> Mapping[str, Any]:
    spec = get_schema(schema_name)
    if not isinstance(value, spec.target_type):
        raise SchemaValidationError(schema_name, f"expected {spec.target_type.__name__}")
    return spec.to_mapping(value)


def _require_fields(schema_name: str, raw: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = tuple(field for field in fields if field not in raw)
    if missing:
        raise SchemaValidationError(schema_name, "missing required field(s): " + ", ".join(missing))


def _string(schema_name: str, value: object, path: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(schema_name, f"{path} must be a string")
    return value


def _number(
    schema_name: str,
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(schema_name, f"{path} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise SchemaValidationError(schema_name, f"{path} must be >= {minimum:g}")
    if maximum is not None and number > maximum:
        raise SchemaValidationError(schema_name, f"{path} must be <= {maximum:g}")
    return number


def _enum(schema_name: str, value: object, path: str, allowed: Sequence[str]) -> str:
    text = _string(schema_name, value, path)
    if text not in allowed:
        raise SchemaValidationError(
            schema_name,
            f"{path} must be one of: " + ", ".join(allowed),
        )
    return text


def _as_mapping(schema_name: str, value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(schema_name, f"{path} must be an object")
    return value


def _array(schema_name: str, value: object, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(schema_name, f"{path} must be an array")
    return value


def _string_array(schema_name: str, value: object, path: str) -> tuple[str, ...]:
    return tuple(_string(schema_name, item, f"{path}[]") for item in _array(schema_name, value, path))


def _from_retrieval_plan(raw: Mapping[str, Any]) -> RetrievalPlan:
    schema = "RetrievalPlan"
    required_facts = tuple(
        _from_required_fact(_as_mapping(schema, item, "required_facts[]"))
        for item in _array(schema, raw.get("required_facts"), "required_facts")
    )
    routes = tuple(
        _from_route(_as_mapping(schema, item, "routes[]"))
        for item in _array(schema, raw.get("routes"), "routes")
    )
    return RetrievalPlan(
        question=_string(schema, raw["question"], "question"),
        required_facts=required_facts,
        routes=routes,
        stop_conditions=_string_array(schema, raw.get("stop_conditions"), "stop_conditions"),
    )


def _from_required_fact(raw: Mapping[str, Any]) -> RequiredFact:
    schema = "RetrievalPlan.required_facts[]"
    _require_fields(schema, raw, ("id", "description", "priority"))
    return RequiredFact(
        id=_string(schema, raw["id"], "id"),
        description=_string(schema, raw["description"], "description"),
        priority=_enum(schema, raw["priority"], "priority", ("must", "should", "nice")),
    )


def _from_route(raw: Mapping[str, Any]) -> Route:
    schema = "RetrievalPlan.routes[]"
    _require_fields(schema, raw, ("fact_id", "candidate_corpus_ids", "reason"))
    return Route(
        fact_id=_string(schema, raw["fact_id"], "fact_id"),
        candidate_corpus_ids=_string_array(schema, raw.get("candidate_corpus_ids"), "candidate_corpus_ids"),
        reason=_string(schema, raw["reason"], "reason"),
    )


def _retrieval_plan_to_mapping(plan: RetrievalPlan) -> Mapping[str, Any]:
    return {
        "question": plan.question,
        "required_facts": [
            {
                "id": fact.id,
                "description": fact.description,
                "priority": fact.priority.value if hasattr(fact.priority, "value") else str(fact.priority),
            }
            for fact in plan.required_facts
        ],
        "routes": [
            {
                "fact_id": route.fact_id,
                "candidate_corpus_ids": list(route.candidate_corpus_ids),
                "reason": route.reason,
            }
            for route in plan.routes
        ],
        "stop_conditions": list(plan.stop_conditions),
    }


def _from_query_rewrite_result(raw: Mapping[str, Any]) -> QueryRewriteResult:
    return QueryRewriteResult(
        subqueries=tuple(
            _from_subquery(_as_mapping("QueryRewriteResult", item, "subqueries[]"))
            for item in _array("QueryRewriteResult", raw.get("subqueries"), "subqueries")
        )
    )


def _from_subquery(raw: Mapping[str, Any]) -> Subquery:
    schema = "QueryRewriteResult.subqueries[]"
    _require_fields(schema, raw, ("id", "fact_id", "query", "target_corpus_ids", "reason"))
    return Subquery(
        id=_string(schema, raw["id"], "id"),
        fact_id=_string(schema, raw["fact_id"], "fact_id"),
        query=_string(schema, raw["query"], "query"),
        target_corpus_ids=_string_array(schema, raw.get("target_corpus_ids"), "target_corpus_ids"),
        reason=_string(schema, raw["reason"], "reason"),
    )


def _query_rewrite_result_to_mapping(result: QueryRewriteResult) -> Mapping[str, Any]:
    return {
        "subqueries": [
            {
                "id": subquery.id,
                "fact_id": subquery.fact_id,
                "query": subquery.query,
                "target_corpus_ids": list(subquery.target_corpus_ids),
                "reason": subquery.reason,
            }
            for subquery in result.subqueries
        ]
    }


def _from_context_assessment(raw: Mapping[str, Any]) -> ContextAssessment:
    return ContextAssessment(
        status=_enum(
            "ContextAssessment",
            raw["status"],
            "status",
            ("sufficient", "insufficient", "irrelevant", "unanswerable"),
        ),
        sufficiency_score=_number(
            "ContextAssessment",
            raw["sufficiency_score"],
            "sufficiency_score",
            minimum=0,
            maximum=1,
        ),
        covered_facts=tuple(
            _from_covered_fact(_as_mapping("ContextAssessment", item, "covered_facts[]"))
            for item in _array("ContextAssessment", raw.get("covered_facts"), "covered_facts")
        ),
        missing_facts=_string_array("ContextAssessment", raw.get("missing_facts"), "missing_facts"),
        unsupported_claims=_string_array("ContextAssessment", raw.get("unsupported_claims"), "unsupported_claims"),
        conflicts=tuple(
            _from_conflict_evidence(_as_mapping("ContextAssessment", item, "conflicts[]"))
            for item in _array("ContextAssessment", raw.get("conflicts", ()), "conflicts")
        ),
        feedback_queries=tuple(
            _from_feedback_query(_as_mapping("ContextAssessment", item, "feedback_queries[]"))
            for item in _array("ContextAssessment", raw.get("feedback_queries"), "feedback_queries")
        ),
        reason=_string("ContextAssessment", raw["reason"], "reason"),
        answerability=_enum(
            "ContextAssessment",
            raw["answerability"],
            "answerability",
            tuple(label.value for label in AnswerabilityLabel),
        )
        if raw.get("answerability") is not None
        else None,
    )


def _from_covered_fact(raw: Mapping[str, Any]) -> CoveredFact:
    schema = "ContextAssessment.covered_facts[]"
    _require_fields(schema, raw, ("fact_id", "snippet_ids"))
    return CoveredFact(
        fact_id=_string(schema, raw["fact_id"], "fact_id"),
        snippet_ids=_string_array(schema, raw.get("snippet_ids"), "snippet_ids"),
    )


def _from_feedback_query(raw: Mapping[str, Any]) -> FeedbackQuery:
    schema = "ContextAssessment.feedback_queries[]"
    _require_fields(schema, raw, ("query", "target_corpus_ids", "reason"))
    return FeedbackQuery(
        query=_string(schema, raw["query"], "query"),
        target_corpus_ids=_string_array(schema, raw.get("target_corpus_ids"), "target_corpus_ids"),
        reason=_string(schema, raw["reason"], "reason"),
        fact_id=_string(schema, raw["fact_id"], "fact_id") if raw.get("fact_id") is not None else None,
    )


def _from_conflict_evidence(raw: Mapping[str, Any]) -> ConflictEvidence:
    schema = "ConflictEvidence"
    _require_fields(schema, raw, ("fact_id", "groups"))
    return ConflictEvidence(
        fact_id=_string(schema, raw["fact_id"], "fact_id"),
        groups=tuple(
            _from_conflicting_evidence_group(_as_mapping(schema, item, "groups[]"))
            for item in _array(schema, raw.get("groups"), "groups")
        ),
        reason=_string(schema, raw["reason"], "reason") if raw.get("reason") is not None else "",
    )


def _from_conflicting_evidence_group(raw: Mapping[str, Any]) -> ConflictingEvidenceGroup:
    schema = "ConflictingEvidenceGroup"
    _require_fields(schema, raw, ("label", "snippet_ids"))
    return ConflictingEvidenceGroup(
        label=_string(schema, raw["label"], "label"),
        snippet_ids=_string_array(schema, raw.get("snippet_ids"), "snippet_ids"),
        value=_string(schema, raw["value"], "value") if raw.get("value") is not None else None,
    )


def _context_assessment_to_mapping(assessment: ContextAssessment) -> Mapping[str, Any]:
    data: dict[str, Any] = {
        "status": assessment.status.value if hasattr(assessment.status, "value") else str(assessment.status),
        "sufficiency_score": assessment.sufficiency_score,
        "covered_facts": [
            {"fact_id": fact.fact_id, "snippet_ids": list(fact.snippet_ids)}
            for fact in assessment.covered_facts
        ],
        "missing_facts": list(assessment.missing_facts),
        "unsupported_claims": list(assessment.unsupported_claims),
        "feedback_queries": [
            _feedback_query_to_mapping(feedback) for feedback in assessment.feedback_queries
        ],
        "reason": assessment.reason,
    }
    if assessment.conflicts:
        data["conflicts"] = [_conflict_evidence_to_mapping(conflict) for conflict in assessment.conflicts]
    if assessment.answerability is not None:
        data["answerability"] = assessment.answerability_label.value
    return data


def _feedback_query_to_mapping(feedback: FeedbackQuery) -> Mapping[str, Any]:
    data: dict[str, Any] = {
        "query": feedback.query,
        "target_corpus_ids": list(feedback.target_corpus_ids),
        "reason": feedback.reason,
    }
    if feedback.fact_id is not None:
        data["fact_id"] = feedback.fact_id
    return data


def _from_grounded_answer(raw: Mapping[str, Any]) -> GroundedAnswer:
    return GroundedAnswer(
        answer=_string("GroundedAnswer", raw["answer"], "answer"),
        citations=tuple(
            _from_grounded_citation(_as_mapping("GroundedAnswer", item, "citations[]"))
            for item in _array("GroundedAnswer", raw.get("citations"), "citations")
        ),
        status=_enum(
            "GroundedAnswer",
            raw["status"],
            "status",
            ("answered", "partial", "unanswerable"),
        ),
        missing_facts=_string_array("GroundedAnswer", raw.get("missing_facts"), "missing_facts"),
        sufficiency_score=_number(
            "GroundedAnswer",
            raw["sufficiency_score"],
            "sufficiency_score",
            minimum=0,
            maximum=1,
        ),
        conflicts=tuple(
            _from_conflict_evidence(_as_mapping("GroundedAnswer", item, "conflicts[]"))
            for item in _array("GroundedAnswer", raw.get("conflicts", ()), "conflicts")
        ),
    )


def _from_grounded_citation(raw: Mapping[str, Any]) -> GroundedCitation:
    schema = "GroundedAnswer.citations[]"
    _require_fields(schema, raw, ("claim", "snippet_ids"))
    return GroundedCitation(
        claim=_string(schema, raw["claim"], "claim"),
        snippet_ids=_string_array(schema, raw.get("snippet_ids"), "snippet_ids"),
    )


def _grounded_answer_to_mapping(answer: GroundedAnswer) -> Mapping[str, Any]:
    data: dict[str, Any] = {
        "answer": answer.answer,
        "citations": [
            {"claim": citation.claim, "snippet_ids": list(citation.snippet_ids)}
            for citation in answer.citations
        ],
        "status": answer.status.value if hasattr(answer.status, "value") else str(answer.status),
        "missing_facts": list(answer.missing_facts),
        "sufficiency_score": answer.sufficiency_score,
    }
    if answer.conflicts:
        data["conflicts"] = [_conflict_evidence_to_mapping(conflict) for conflict in answer.conflicts]
    return data


def _conflict_evidence_to_mapping(conflict: ConflictEvidence) -> Mapping[str, Any]:
    return {
        "fact_id": conflict.fact_id,
        "groups": [
            {
                key: value
                for key, value in {
                    "label": group.label,
                    "snippet_ids": list(group.snippet_ids),
                    "value": group.value,
                }.items()
                if value is not None
            }
            for group in conflict.groups
        ],
        "reason": conflict.reason,
    }


SCHEMA_REGISTRY: Mapping[str, SchemaSpec] = {
    "RetrievalPlan": SchemaSpec(
        name="RetrievalPlan",
        target_type=RetrievalPlan,
        required_fields=("question", "required_facts", "routes", "stop_conditions"),
        from_mapping=_from_retrieval_plan,
        to_mapping=_retrieval_plan_to_mapping,
    ),
    "QueryRewriteResult": SchemaSpec(
        name="QueryRewriteResult",
        target_type=QueryRewriteResult,
        required_fields=("subqueries",),
        from_mapping=_from_query_rewrite_result,
        to_mapping=_query_rewrite_result_to_mapping,
    ),
    "ContextAssessment": SchemaSpec(
        name="ContextAssessment",
        target_type=ContextAssessment,
        required_fields=(
            "status",
            "sufficiency_score",
            "covered_facts",
            "missing_facts",
            "unsupported_claims",
            "feedback_queries",
            "reason",
        ),
        from_mapping=_from_context_assessment,
        to_mapping=_context_assessment_to_mapping,
    ),
    "GroundedAnswer": SchemaSpec(
        name="GroundedAnswer",
        target_type=GroundedAnswer,
        required_fields=("answer", "citations", "status", "missing_facts", "sufficiency_score"),
        from_mapping=_from_grounded_answer,
        to_mapping=_grounded_answer_to_mapping,
    ),
}
