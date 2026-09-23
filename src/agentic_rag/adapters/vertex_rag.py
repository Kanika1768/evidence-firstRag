"""Optional Vertex AI RAG Engine cross-corpus retriever adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentic_rag.contracts import CorpusDescriptor, RetrievalHit, SubQuery


@dataclass(frozen=True)
class VertexRagConfig:
    project_id: str
    location: str
    rag_corpus_resource_names: tuple[str, ...]


class VertexRagCrossCorpusRetriever:
    """Retriever adapter for Google RAG Engine cross-corpus retrieval.

    The implementation intentionally keeps dependencies optional so core package
    usage does not require Google AI SDK packages.
    """

    def __init__(self, config: VertexRagConfig) -> None:
        self.config = config

    def retrieve(
        self,
        subqueries: Sequence[SubQuery],
        corpora: Sequence[CorpusDescriptor],
    ) -> Sequence[RetrievalHit]:
        if not subqueries:
            return ()
        try:
            from vertexai.preview import rag
            import vertexai
        except ImportError as exc:
            raise RuntimeError(
                "Vertex RAG retrieval requires the optional Google Cloud Vertex AI SDK. "
                "Install and authenticate it in the target project before using this adapter."
            ) from exc

        vertexai.init(project=self.config.project_id, location=self.config.location)
        corpus_map = _corpus_resource_map(self.config, corpora)

        hits: list[RetrievalHit] = []
        for subquery in subqueries:
            resources = [
                rag.RagResource(rag_corpus=corpus_map[corpus_id])
                for corpus_id in subquery.target_corpus_ids
                if corpus_id in corpus_map
            ]
            if not resources:
                continue
            response = rag.retrieval_query(
                rag_resources=resources,
                text=subquery.query,
            )
            hits.extend(_response_to_hits(response, subquery))
        return tuple(hits)


def _corpus_resource_map(
    config: VertexRagConfig,
    corpora: Sequence[CorpusDescriptor],
) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for corpus in corpora:
        resource_name = corpus.metadata.get("rag_corpus_resource_name")
        if isinstance(resource_name, str):
            by_id[corpus.id] = resource_name
    for corpus_id, resource_name in zip(
        (corpus.id for corpus in corpora if corpus.id not in by_id),
        config.rag_corpus_resource_names,
    ):
        by_id[corpus_id] = resource_name
    return by_id


def _response_to_hits(response: object, subquery: SubQuery) -> list[RetrievalHit]:
    contexts = getattr(response, "contexts", None)
    raw_contexts = getattr(contexts, "contexts", ()) if contexts is not None else ()
    hits: list[RetrievalHit] = []
    for index, context in enumerate(raw_contexts, start=1):
        text = getattr(context, "text", "")
        source_uri = getattr(context, "source_uri", "")
        document_id = source_uri or getattr(context, "chunk", f"context-{index}")
        corpus_id = subquery.target_corpus_ids[0] if subquery.target_corpus_ids else "vertex-rag"
        score = getattr(context, "score", 0.0)
        score_value = float(score) if isinstance(score, (int, float)) else 0.0
        hits.append(
            RetrievalHit(
                id=f"{subquery.id}:vertex:{index}",
                corpus_id=corpus_id,
                document_id=str(document_id),
                text=str(text),
                score=score_value,
                metadata={"subquery_id": subquery.id, "fact_id": subquery.fact_id, "source_uri": source_uri},
            )
        )
    return hits
