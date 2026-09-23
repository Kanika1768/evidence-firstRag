"""Stable contracts for the portable Agentic RAG loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Protocol, Sequence


class FactPriority(str, Enum):
    MUST = "must"
    SHOULD = "should"
    NICE = "nice"


AssessmentStatus = Literal["sufficient", "insufficient", "irrelevant", "unanswerable"]


class ContextStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    IRRELEVANT = "irrelevant"
    UNANSWERABLE = "unanswerable"


class AnswerabilityLabel(str, Enum):
    SUFFICIENT = "sufficient"
    USEFUL_BUT_INCOMPLETE = "useful_but_incomplete"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    UNANSWERABLE = "unanswerable"


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    PARTIAL = "partial"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class Corpus:
    id: str
    description: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredFact:
    id: str
    description: str
    priority: FactPriority | str = FactPriority.MUST
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_must(self) -> bool:
        priority = self.priority.value if isinstance(self.priority, FactPriority) else str(self.priority)
        return priority == FactPriority.MUST.value


@dataclass(frozen=True)
class Route:
    fact_id: str
    candidate_corpus_ids: Sequence[str]
    reason: str


@dataclass(frozen=True)
class RetrievalPlan:
    question: str
    required_facts: Sequence[RequiredFact]
    routes: Sequence[Route]
    stop_conditions: Sequence[str] = field(default_factory=tuple)

    def route_for_fact(self, fact_id: str) -> Route | None:
        for route in self.routes:
            if route.fact_id == fact_id:
                return route
        return None


@dataclass(frozen=True)
class Subquery:
    id: str
    fact_id: str
    query: str
    target_corpus_ids: Sequence[str]
    reason: str
    parent_query: str | None = None
    iteration: int = 0


@dataclass(frozen=True)
class QueryRewriteResult:
    subqueries: Sequence[Subquery] = field(default_factory=tuple)


@dataclass(frozen=True)
class Snippet:
    id: str
    corpus_id: str
    document_id: str
    text: str
    score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    span: tuple[int, int] | None = None
    query_id: str | None = None
    fact_id: str | None = None


@dataclass(frozen=True)
class Claim:
    text: str
    snippet_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class DraftAnswer:
    """Draft used by both legacy and current callers.

    Legacy callers in earlier revisions used ``text`` + ``cited_snippet_ids`` while
    the newer pipeline uses ``claims``. Both inputs are supported and normalized.
    """

    text: str = ""
    claims: Sequence[Claim] = field(default_factory=tuple)
    cited_snippet_ids: Sequence[str] = field(default_factory=tuple)
    unsupported_claims: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized_text = self.text.strip() if isinstance(self.text, str) else ""
        normalized_claims = tuple(
            claim if isinstance(claim, Claim) else Claim(str(claim), tuple())
            for claim in self.claims
        )
        normalized_cited = tuple(
            str(snippet_id)
            for snippet_id in self.cited_snippet_ids
        )
        normalized_unsupported = tuple(str(item) for item in self.unsupported_claims)

        if not normalized_claims and normalized_text and normalized_cited:
            normalized_claims = (Claim(normalized_text, normalized_cited),)
        elif not normalized_claims and normalized_text:
            normalized_claims = (Claim(normalized_text, ()),)

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "claims", normalized_claims)
        object.__setattr__(self, "cited_snippet_ids", normalized_cited)
        object.__setattr__(self, "unsupported_claims", normalized_unsupported)


@dataclass(frozen=True)
class CoveredFact:
    fact_id: str
    snippet_ids: Sequence[str]


@dataclass(frozen=True)
class ConflictingEvidenceGroup:
    label: str
    snippet_ids: Sequence[str]
    value: str | None = None


@dataclass(frozen=True)
class ConflictEvidence:
    fact_id: str
    groups: Sequence[ConflictingEvidenceGroup]
    reason: str = ""


@dataclass(frozen=True)
class FeedbackQuery:
    query: str
    target_corpus_ids: Sequence[str]
    reason: str
    fact_id: str | None = None


@dataclass(frozen=True)
class ContextAssessment:
    status: ContextStatus | str
    sufficiency_score: float
    covered_facts: Sequence[CoveredFact] = field(default_factory=tuple)
    missing_facts: Sequence[str] = field(default_factory=tuple)
    unsupported_claims: Sequence[str] = field(default_factory=tuple)
    conflicts: Sequence[ConflictEvidence] = field(default_factory=tuple)
    feedback_queries: Sequence[FeedbackQuery] = field(default_factory=tuple)
    reason: str = ""
    answerability: AnswerabilityLabel | str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.sufficiency_score <= 1:
            raise ValueError("sufficiency_score must be between 0 and 1")
        if self.answerability is not None:
            AnswerabilityLabel(_enum_value(self.answerability))

    @property
    def confidence(self) -> float:
        """Compatibility alias for schemas that name this score confidence."""
        return self.sufficiency_score

    @property
    def answerability_label(self) -> AnswerabilityLabel:
        if self.answerability is not None:
            return AnswerabilityLabel(_enum_value(self.answerability))
        status = _enum_value(self.status)
        if status == ContextStatus.SUFFICIENT.value:
            return AnswerabilityLabel.SUFFICIENT
        if status == ContextStatus.UNANSWERABLE.value:
            return AnswerabilityLabel.UNANSWERABLE
        if status == ContextStatus.IRRELEVANT.value:
            return AnswerabilityLabel.UNANSWERABLE
        return AnswerabilityLabel.INSUFFICIENT


@dataclass(frozen=True)
class GroundedCitation:
    claim: str
    snippet_ids: Sequence[str]


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: Sequence[GroundedCitation]
    status: AnswerStatus | str
    missing_facts: Sequence[str] = field(default_factory=tuple)
    sufficiency_score: float = 0.0
    conflicts: Sequence[ConflictEvidence] = field(default_factory=tuple)
    iterations: int = 0
    audit: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationTrace:
    iteration: int
    subqueries: Sequence[Subquery]
    snippets: Sequence[Snippet]
    draft: DraftAnswer
    assessment: ContextAssessment


@dataclass(frozen=True)
class RunResult:
    question: str
    plan: RetrievalPlan
    answer: GroundedAnswer
    iterations: Sequence[IterationTrace]
    snippets: Sequence[Snippet]


# Compatibility dataclass names and aliases from earlier package generations.
@dataclass(frozen=True)
class IterationState:
    question: str
    iteration: int = 0
    plan: RetrievalPlan | None = None
    subqueries: Sequence[Subquery] = field(default_factory=tuple)
    hits: Sequence[Snippet] = field(default_factory=tuple)
    draft: DraftAnswer | None = None
    assessment: ContextAssessment | None = None
    prior_feedback: Sequence[FeedbackQuery] = field(default_factory=tuple)


CorpusDescriptor = Corpus
RetrievalRoute = Route
SubQuery = Subquery
RetrievalHit = Snippet
ClaimCitation = GroundedCitation


class Planner(Protocol):
    def plan(
        self,
        question: str,
        corpus_catalog: Sequence[Corpus],
        prior_assessment: ContextAssessment | None = None,
    ) -> RetrievalPlan:
        ...


class QueryRewriter(Protocol):
    def rewrite(
        self,
        question: str,
        plan: RetrievalPlan,
        prior_assessment: ContextAssessment | None,
        iteration: int,
    ) -> Sequence[Subquery]:
        ...


class Retriever(Protocol):
    def retrieve(self, subquery: Subquery) -> Sequence[Snippet]:
        ...


class Drafter(Protocol):
    def draft(self, question: str, plan: RetrievalPlan, snippets: Sequence[Snippet]) -> DraftAnswer:
        ...


class SufficientContextJudge(Protocol):
    def assess(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        draft: DraftAnswer,
    ) -> ContextAssessment:
        ...


class Synthesizer(Protocol):
    def synthesize(
        self,
        question: str,
        plan: RetrievalPlan,
        snippets: Sequence[Snippet],
        assessment: ContextAssessment,
    ) -> GroundedAnswer:
        ...


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)
