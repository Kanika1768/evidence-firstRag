"""General-purpose deterministic planner for document-grounded RAG."""

from __future__ import annotations

import re
from typing import Sequence

from .contracts import (
    Corpus,
    ContextAssessment,
    FactPriority,
    FeedbackQuery,
    RequiredFact,
    RetrievalPlan,
    Route,
)


class GeneralPlanner:
    """Create retrieval plans for arbitrary questions and document corpora."""

    def plan(
        self,
        question: str,
        corpus_catalog: Sequence[Corpus],
        prior_assessment: ContextAssessment | None = None,
    ) -> RetrievalPlan:
        question = question.strip()

        if not question:
            raise ValueError("question must not be empty")

        facts = self._extract_required_facts(question, prior_assessment)

        routes = tuple(
            Route(
                fact_id=fact.id,
                candidate_corpus_ids=self._rank_corpora(
                    fact.description,
                    corpus_catalog,
                ),
                reason=(
                    "Routed using lexical overlap between the required "
                    "fact and document description."
                ),
            )
            for fact in facts
        )

        return RetrievalPlan(
            question=question,
            required_facts=tuple(facts),
            routes=routes,
            stop_conditions=(
                "All must-have facts have supporting evidence.",
                "No targeted recovery queries remain.",
                "Maximum iteration budget is reached.",
            ),
        )

    # Compatibility with the older AgenticRAGPipeline API.
    def create_plan(
        self,
        question: str,
        corpora: Sequence[Corpus],
        prior_feedback: Sequence[FeedbackQuery] = (),
    ) -> RetrievalPlan:
        facts = self._extract_required_facts(question, None)

        existing = {_normalize(fact.description) for fact in facts}

        for feedback in prior_feedback:
            normalized = _normalize(feedback.query)

            if normalized and normalized not in existing:
                facts.append(
                    RequiredFact(
                        id=f"f{len(facts) + 1}",
                        description=feedback.query,
                        priority=FactPriority.MUST,
                    )
                )
                existing.add(normalized)

        routes = tuple(
            Route(
                fact_id=fact.id,
                candidate_corpus_ids=self._rank_corpora(
                    fact.description,
                    corpora,
                ),
                reason="Initial or recovery routing based on lexical overlap.",
            )
            for fact in facts
        )

        return RetrievalPlan(
            question=question,
            required_facts=tuple(facts),
            routes=routes,
            stop_conditions=(
                "All must-have facts have supporting evidence.",
                "No targeted recovery queries remain.",
            ),
        )

    def _extract_required_facts(
        self,
        question: str,
        prior_assessment: ContextAssessment | None,
    ) -> list[RequiredFact]:
        facts: list[str] = []

        if prior_assessment is not None:
            facts.extend(prior_assessment.missing_facts)

        question = question.strip().rstrip("?")

        parts = self._split_question(question)

        facts.extend(parts)

        deduplicated: list[str] = []
        seen: set[str] = set()

        for fact in facts:
            normalized = _normalize(fact)

            if normalized and normalized not in seen:
                seen.add(normalized)
                deduplicated.append(fact.strip())

        return [
            RequiredFact(
                id=f"f{index}",
                description=fact,
                priority=FactPriority.MUST,
                metadata={
                    "required_terms": _meaningful_terms(fact),
                },
            )
            for index, fact in enumerate(deduplicated, start=1)
        ]

    def _split_question(self, question: str) -> list[str]:
        """Split common multi-part questions while preserving the subject."""

        # No conjunction → the complete question is one retrieval requirement.
        if not re.search(r"\b(and|also|as well as)\b", question, re.IGNORECASE):
            return [question]

        # Identify the main entity/subject from the first clause.
        #
        # Example:
        #   Who founded NovaTech and when was it founded
        #
        # subject = NovaTech
        first_clause = re.split(
            r"\b(?:and|also|as well as)\b",
            question,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()

        second_clause = re.split(
            r"\b(?:and|also|as well as)\b",
            question,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[1].strip()

        subject = self._extract_subject(first_clause)

        first_fact = first_clause

        if subject and subject.lower() not in second_clause.lower():
            second_fact = f"{second_clause} about {subject}"
        else:
            second_fact = second_clause

        return [first_fact, second_fact]

    def _extract_subject(self, clause: str) -> str | None:
        """Extract a likely named entity from a question clause."""

        # Prefer a capitalized multi-word/name token.
        matches = re.findall(
            r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)*\b",
            clause,
        )

        ignored = {
            "Who",
            "What",
            "When",
            "Where",
            "Why",
            "How",
            "Which",
            "Did",
            "Does",
            "Is",
            "Are",
            "Was",
            "Were",
        }

        for match in reversed(matches):
            if match not in ignored:
                return match

        return None

    def _rank_corpora(
        self,
        fact: str,
        corpora: Sequence[Corpus],
    ) -> tuple[str, ...]:
        fact_tokens = _tokens(fact)

        scored: list[tuple[float, str]] = []

        for corpus in corpora:
            corpus_text = f"{corpus.id} {corpus.description}"
            corpus_tokens = _tokens(corpus_text)

            if not fact_tokens:
                score = 0.0
            else:
                overlap = fact_tokens & corpus_tokens
                score = len(overlap) / len(fact_tokens)

            scored.append((score, corpus.id))

        scored.sort(key=lambda item: (-item[0], item[1]))

        relevant = tuple(
            corpus_id
            for score, corpus_id in scored
            if score > 0
        )

        # Never discard the whole corpus when lexical routing has no match.
        if not relevant:
            return tuple(corpus.id for corpus in corpora)

        return relevant


def _meaningful_terms(text: str) -> tuple[str, ...]:
    """Extract content-bearing terms for evidence matching."""

    stopwords = {
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "which",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "will",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
        "and",
        "or",
        "it",
        "its",
        "about",
        "of",
        "to",
        "for",
        "in",
        "on",
        "by",
        "with",
    }

    terms = []
    seen = set()

    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in stopwords:
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)

    return tuple(terms)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))
