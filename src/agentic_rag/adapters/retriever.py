"""Dependency-free retriever adapters with preserved snippet provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from agentic_rag.contracts import Snippet, Subquery


LEXICAL_SCORING_RULE = (
    "Score = number of unique overlapping query terms plus 2.0 for an exact phrase match; "
    "results are ordered by score descending, then corpus_id, document_id, and snippet id."
)


@dataclass(frozen=True)
class LexicalDocument:
    corpus_id: str
    document_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _TokenMatch:
    token: str
    start: int
    end: int


class LexicalRetriever:
    """Simple lexical retriever for local tests, demos, and adapter baselines."""

    def __init__(self, documents: Sequence[LexicalDocument], *, per_query_limit: int = 5) -> None:
        self.documents = tuple(documents)
        self.per_query_limit = per_query_limit
        self.calls: list[Subquery] = []

    def retrieve(self, subquery: Subquery) -> Sequence[Snippet]:
        self.calls.append(subquery)
        query_terms = _tokens(subquery.query)
        if not query_terms:
            return ()

        matches_by_document: dict[tuple[str, str], Snippet] = {}
        for document in self.documents:
            if document.corpus_id not in subquery.target_corpus_ids:
                continue
            token_matches = _token_matches(document.text)
            document_terms = {match.token for match in token_matches}
            overlap = query_terms & document_terms
            phrase_span = _phrase_span(document.text, subquery.query)
            if not overlap and phrase_span is None:
                continue

            score = float(len(overlap)) + (2.0 if phrase_span is not None else 0.0)
            span = phrase_span or _overlap_span(token_matches, overlap)
            snippet = Snippet(
                id=f"{document.corpus_id}:{document.document_id}",
                corpus_id=document.corpus_id,
                document_id=document.document_id,
                text=document.text,
                score=score,
                metadata=document.metadata,
                span=span,
                query_id=subquery.id,
                fact_id=subquery.fact_id,
            )
            document_key = (document.corpus_id, document.document_id)
            existing = matches_by_document.get(document_key)
            if existing is None or _sort_key(snippet) < _sort_key(existing):
                matches_by_document[document_key] = snippet

        return tuple(sorted(matches_by_document.values(), key=_sort_key)[: self.per_query_limit])


def _sort_key(snippet: Snippet) -> tuple[float, str, str, str]:
    return (-snippet.score, snippet.corpus_id, snippet.document_id, snippet.id)


def _normalize_token(token: str) -> str:
    """Normalize simple English word variants for lexical matching."""
    token = token.lower()

    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[:-len(suffix)]

    return token


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(match.group(0))
        for match in re.finditer(r"[a-z0-9]+", text.lower())
    }


def _token_matches(text: str) -> tuple[_TokenMatch, ...]:
    return tuple(
        _TokenMatch(
            _normalize_token(match.group(0)),
            match.start(),
            match.end(),
        )
        for match in re.finditer(r"[a-z0-9]+", text.lower())
    )


def _phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    if not phrase.strip():
        return None
    start = text.lower().find(phrase.lower())
    if start < 0:
        return None
    return (start, start + len(phrase))


def _overlap_span(token_matches: Sequence[_TokenMatch], overlap: set[str]) -> tuple[int, int] | None:
    overlapping = tuple(match for match in token_matches if match.token in overlap)
    if not overlapping:
        return None
    return (overlapping[0].start, overlapping[-1].end)
