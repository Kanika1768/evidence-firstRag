"""Hybrid Retrieval Pipeline for EvidenceFirst RAG (Person 1 — Avni).

Dense embeddings (sentence-transformers + FAISS) and Okapi BM25 run in
parallel, their rankings are fused with Reciprocal Rank Fusion, and the fused
candidates are reranked by a cross-encoder to return the final top 6 chunks.

When optional dependencies are missing (``pip install -e .[retrieval]``), each
stage falls back to a deterministic pure-Python implementation so the package
and its tests still run: feature-hashed embeddings, a brute-force inner-product
index, and a lexical reranker. ``backend_name`` on each component reports what
is actually in use, and ``EVIDENCEFIRST_DISABLE_NEURAL=1`` forces the fallbacks.

Also provides the retrieval evaluation used for the ablation table
(Recall@5, Recall@10, MRR for dense-only, BM25-only, hybrid, hybrid+reranker).
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from evidencefirst_rag.chunking import _TOKEN_RE, Chunk

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
RRF_K = 60
TOP_K_EVIDENCE = 6
CANDIDATE_POOL = 50
RERANK_POOL = 30
ENCODER_WINDOW_TOKENS = 300
ENCODER_WINDOW_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RETRIEVAL_MODES = ("dense_only", "bm25_only", "hybrid", "hybrid_rerank")
MODE_LABELS = {
    "dense_only": "Dense-only (FAISS)",
    "bm25_only": "BM25-only (Okapi)",
    "hybrid": "Hybrid (RRF Fusion)",
    "hybrid_rerank": "Hybrid + Reranker (Cross-Encoder)",
}

_EMBED_CACHE: dict[tuple[str, str], list[float]] = {}


def _neural_disabled() -> bool:
    return os.environ.get("EVIDENCEFIRST_DISABLE_NEURAL", "").lower() in {"1", "true", "yes"}


@functools.lru_cache(maxsize=None)
def _load_sentence_transformer(model_name: str):
    if _neural_disabled():
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning("Could not load embedding model %s (%s); using feature-hash fallback.", model_name, exc)
        return None


@functools.lru_cache(maxsize=None)
def _load_cross_encoder(model_name: str):
    if _neural_disabled():
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        return CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        logger.warning("Could not load reranker %s (%s); using lexical fallback.", model_name, exc)
        return None


def _section_prefix(chunk: Chunk) -> str:
    if chunk.section and chunk.section not in {chunk.document_id, "General"}:
        return f"{chunk.section}\n"
    return ""


def index_text(chunk: Chunk) -> str:
    """Text that is keyword-indexed for a chunk: its section heading plus its body."""
    return _section_prefix(chunk) + chunk.text


def encoder_windows(chunk: Chunk) -> list[str]:
    """Split a chunk into overlapping windows short enough for 512-token neural encoders.

    A 550-token chunk is often 600-800 word pieces, beyond what bge-small or the
    MiniLM cross-encoder read, so the tail of the chunk would be silently truncated.
    Each window keeps the section heading; a chunk's dense / rerank score is the
    best score over its windows.
    """
    spans = [m.span() for m in _TOKEN_RE.finditer(chunk.text)]
    if len(spans) <= ENCODER_WINDOW_TOKENS:
        return [index_text(chunk)]
    prefix = _section_prefix(chunk)
    step = ENCODER_WINDOW_TOKENS - ENCODER_WINDOW_OVERLAP
    windows = []
    for start in range(0, len(spans), step):
        end = min(len(spans), start + ENCODER_WINDOW_TOKENS)
        windows.append(prefix + chunk.text[spans[start][0] : spans[end - 1][1]])
        if end >= len(spans):
            break
    return windows


def _best_per_chunk(hits: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
    """Collapse window-level hits (sorted by score) to one entry per chunk id, keeping the best."""
    seen: set[str] = set()
    best = []
    for cid, score in hits:
        if cid not in seen:
            seen.add(cid)
            best.append((cid, score))
    return best


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# 1. Dense Embedding Model & FAISS Vector Index
# ---------------------------------------------------------------------------

class DenseEmbeddingModel:
    """L2-normalized sentence embeddings (384-dim for the default bge-small model)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._st_model = _load_sentence_transformer(model_name)
        self.dim = EMBEDDING_DIM
        if self._st_model is not None:
            get_dim = getattr(self._st_model, "get_embedding_dimension", None) or self._st_model.get_sentence_embedding_dimension
            self.dim = get_dim()

    @property
    def backend_name(self) -> str:
        return f"sentence-transformers:{self.model_name}" if self._st_model is not None else "feature-hash-fallback"

    def encode(self, text: str) -> list[float]:
        """Embed a passage (chunk text)."""
        return self.encode_many([text])[0]

    def encode_query(self, query: str) -> list[float]:
        """Embed a search query (bge models expect an instruction prefix on queries only)."""
        if self._st_model is not None and "bge" in self.model_name.lower():
            query = BGE_QUERY_INSTRUCTION + query
        if self._st_model is not None:
            return [float(x) for x in self._st_model.encode(query, convert_to_numpy=True, normalize_embeddings=True)]
        return self._hash_embed(query)

    def encode_many(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Embed many passages, reusing cached vectors for texts already seen in this process."""
        backend = self.backend_name
        missing = list(dict.fromkeys(t for t in texts if (backend, t) not in _EMBED_CACHE))
        if missing:
            if self._st_model is not None:
                matrix = self._st_model.encode(
                    missing, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
                )
                vectors = [[float(x) for x in row] for row in matrix]
            else:
                vectors = [self._hash_embed(t) for t in missing]
            for text, vec in zip(missing, vectors):
                _EMBED_CACHE[(backend, text)] = vec
        return [_EMBED_CACHE[(backend, t)] for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        """Reproducible feature-hashed embedding over words and character 3/4-grams, L2-normalized."""
        vec = [0.0] * self.dim
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vec

        for pos, token in enumerate(tokens):
            pos_weight = 1.0 / (1.0 + 0.05 * math.log(pos + 1))
            h1 = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            vec[h1 % self.dim] += (1.0 if (h1 >> 8) & 1 else -1.0) * 1.5 * pos_weight
            for n in (3, 4):
                for i in range(len(token) - n + 1):
                    h_ng = int(hashlib.md5(token[i : i + n].encode("utf-8")).hexdigest()[:8], 16)
                    vec[h_ng % self.dim] += (1.0 if (h_ng >> 8) & 1 else -1.0) * 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm > 0.0 else vec


class FAISSVectorIndex:
    """Inner-product (cosine on normalized vectors) index backed by ``faiss.IndexFlatIP`` when available."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.chunk_ids: list[str] = []
        self.vectors: list[list[float]] = []
        self._faiss_index = None
        try:
            import faiss

            self._faiss_index = faiss.IndexFlatIP(dim)
        except ImportError:
            self._faiss_index = None

    @property
    def backend_name(self) -> str:
        return "FAISS IndexFlatIP" if self._faiss_index is not None else "pure-python-dotproduct"

    def add(self, chunk_id: str, vector: list[float]) -> None:
        self.add_many([chunk_id], [vector])

    def add_many(self, chunk_ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if not chunk_ids:
            return
        self.chunk_ids.extend(chunk_ids)
        if self._faiss_index is not None:
            import numpy as np

            self._faiss_index.add(np.asarray(vectors, dtype=np.float32))
        else:
            self.vectors.extend(list(v) for v in vectors)

    def search(self, query_vector: list[float], top_k: int = 20) -> list[tuple[str, float]]:
        """Return the ``top_k`` (chunk_id, score) pairs by inner product."""
        if not self.chunk_ids:
            return []

        if self._faiss_index is not None:
            import numpy as np

            k = min(top_k, len(self.chunk_ids))
            scores, indices = self._faiss_index.search(np.asarray([query_vector], dtype=np.float32), k)
            return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], indices[0]) if 0 <= i < len(self.chunk_ids)]

        scored = [(cid, sum(q * v for q, v in zip(query_vector, vec))) for cid, vec in zip(self.chunk_ids, self.vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        """Write a native FAISS index (or a JSON vector file without FAISS) plus ``<path>.ids.json``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._faiss_index is not None:
            import faiss

            faiss.write_index(self._faiss_index, str(path))
            fmt = "faiss"
        else:
            path.write_text(json.dumps({"vectors": self.vectors}), encoding="utf-8")
            fmt = "json"
        sidecar = {"format": fmt, "dim": self.dim, "chunk_ids": self.chunk_ids}
        Path(f"{path}.ids.json").write_text(json.dumps(sidecar), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> FAISSVectorIndex:
        path = Path(path)
        sidecar = json.loads(Path(f"{path}.ids.json").read_text(encoding="utf-8"))
        index = cls(dim=int(sidecar["dim"]))
        if sidecar["format"] == "faiss":
            if index._faiss_index is None:
                raise RuntimeError(f"{path} is a native FAISS index; install faiss-cpu to load it.")
            import faiss

            index._faiss_index = faiss.read_index(str(path))
            index.chunk_ids = list(sidecar["chunk_ids"])
        else:
            vectors = json.loads(path.read_text(encoding="utf-8"))["vectors"]
            index.add_many(sidecar["chunk_ids"], vectors)
        if len(index.chunk_ids) != (index._faiss_index.ntotal if index._faiss_index is not None else len(index.vectors)):
            raise RuntimeError(f"{path} and its id sidecar disagree on the number of vectors.")
        return index


# ---------------------------------------------------------------------------
# 2. Okapi BM25 Keyword Search Index
# ---------------------------------------------------------------------------

class BM25Index:
    """Okapi BM25 keyword index (k1=1.5, b=0.75) with non-negative Robertson-Spärck Jones IDF."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_len = 0.0
        self.inverted_index: dict[str, dict[str, int]] = {}
        self.doc_count = 0
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 1]

    def fit(self, chunks: Sequence[Chunk]) -> None:
        self.fit_texts([(c.chunk_id, index_text(c)) for c in chunks])

    def fit_texts(self, docs: Sequence[tuple[str, str]]) -> None:
        self.doc_count = len(docs)
        self.doc_lengths.clear()
        self.inverted_index.clear()
        self.idf.clear()
        if self.doc_count == 0:
            return

        total_tokens = 0
        for doc_id, text in docs:
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term, count in tf.items():
                self.inverted_index.setdefault(term, {})[doc_id] = count

        self.avg_doc_len = total_tokens / self.doc_count
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or self.doc_count == 0:
            return []

        scores: dict[str, float] = {}
        for token in q_tokens:
            idf_val = self.idf.get(token)
            if idf_val is None:
                continue
            for doc_id, tf in self.inverted_index[token].items():
                dl = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avg_doc_len or 1.0)))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf_val * (tf * (self.k1 + 1.0)) / denom

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(*rankings: Sequence[tuple[str, float]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse rankings with RRF: score(d) = sum over rankings of 1 / (k + rank(d)).

    Ties are broken by first appearance, which keeps the fusion deterministic.
    """
    rrf_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _) in enumerate(ranking, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# 4. Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly with a cross-encoder and keeps the best ``top_k``."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._st_reranker = _load_cross_encoder(model_name)

    @property
    def backend_name(self) -> str:
        return f"cross-encoder:{self.model_name}" if self._st_reranker is not None else "lexical-token-overlap-fallback"

    def rerank(self, query: str, candidates: Sequence[Chunk], top_k: int = TOP_K_EVIDENCE) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []

        if self._st_reranker is not None:
            owners: list[int] = []
            pairs: list[tuple[str, str]] = []
            for i, chunk in enumerate(candidates):
                for window in encoder_windows(chunk):
                    owners.append(i)
                    pairs.append((query, window))
            best = [-math.inf] * len(candidates)
            for owner, score in zip(owners, self._st_reranker.predict(pairs, batch_size=32)):
                best[owner] = max(best[owner], float(score))
            scored = list(zip(candidates, best))
        else:
            scored = [(c, self._lexical_score(query, index_text(c))) for c in candidates]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        c_text = text.lower()
        c_tokens = re.findall(r"[a-z0-9]+", c_text)
        overlap_ratio = len(q_tokens & set(c_tokens)) / len(q_tokens) if q_tokens else 0.0
        phrase_bonus = 2.0 if query.lower().strip() in c_text else 0.0
        early_bonus = 0.05 * sum(1 for tok in c_tokens[:100] if tok in q_tokens)
        return overlap_ratio * 3.0 + phrase_bonus + early_bonus


def build_vector_index(chunks: Sequence[Chunk], embedder: DenseEmbeddingModel) -> FAISSVectorIndex:
    """Embed every encoder window of every chunk; the index may hold several vectors per chunk id."""
    ids: list[str] = []
    texts: list[str] = []
    for chunk in chunks:
        for window in encoder_windows(chunk):
            ids.append(chunk.chunk_id)
            texts.append(window)
    index = FAISSVectorIndex(dim=embedder.dim)
    index.add_many(ids, embedder.encode_many(texts))
    return index


# ---------------------------------------------------------------------------
# 5. Hybrid Retrieval Manager
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Dense + BM25 -> RRF -> cross-encoder -> top 6 evidence chunks."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedding_model: DenseEmbeddingModel | None = None,
        reranker: CrossEncoderReranker | None = None,
        vector_index: FAISSVectorIndex | None = None,
        candidate_pool: int = CANDIDATE_POOL,
        rerank_pool: int = RERANK_POOL,
    ) -> None:
        self.chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.embedding_model = embedding_model or DenseEmbeddingModel()
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_pool = candidate_pool
        self.rerank_pool = rerank_pool

        self.bm25_index = BM25Index()
        self.bm25_index.fit(list(self.chunks_by_id.values()))

        if vector_index is not None:
            if set(vector_index.chunk_ids) != set(self.chunks_by_id):
                raise ValueError("vector_index chunk ids do not match the supplied chunks")
            self.vector_index = vector_index
        else:
            self.vector_index = build_vector_index(list(self.chunks_by_id.values()), self.embedding_model)

    @classmethod
    def from_processed(cls, directory: Path, **kwargs) -> HybridRetriever:
        """Load ``chunks`` and the prebuilt FAISS index written by ``scripts/ingest.py``.

        The saved index is reused only when the current embedding backend and encoder
        window settings match ``manifest.json``; otherwise vectors are recomputed.
        """
        directory = Path(directory)
        chunks = load_processed_chunks(directory)
        embedder = kwargs.pop("embedding_model", None) or DenseEmbeddingModel()
        manifest_path = directory / "manifest.json"
        vector_index = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            windows = {"tokens": ENCODER_WINDOW_TOKENS, "overlap": ENCODER_WINDOW_OVERLAP}
            if manifest.get("embedding_backend") == embedder.backend_name and manifest.get("encoder_windows") == windows:
                try:
                    vector_index = FAISSVectorIndex.load(directory / "faiss.index")
                except RuntimeError as exc:
                    logger.warning("Rebuilding dense index: %s", exc)
        return cls(chunks, embedding_model=embedder, vector_index=vector_index, **kwargs)

    def _ranked(self, hits: Iterable[tuple[str, float]], top_k: int, exclude: set[str]) -> list[Chunk]:
        results: list[Chunk] = []
        seen_texts: set[str] = set()
        for cid, _ in hits:
            chunk = self.chunks_by_id.get(cid)
            if chunk is None or cid in exclude:
                continue
            key = _normalize_for_dedup(chunk.text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            results.append(chunk)
            if len(results) >= top_k:
                break
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_EVIDENCE,
        mode: str = "hybrid_rerank",
        exclude_chunk_ids: Iterable[str] = (),
    ) -> list[Chunk]:
        """Return up to ``top_k`` chunks for ``query``.

        Modes: ``dense_only``, ``bm25_only``, ``hybrid`` (RRF), ``hybrid_rerank`` (RRF + cross-encoder).
        ``exclude_chunk_ids`` drops chunks already in hand (e.g. during recovery retrieval);
        chunks with identical text are returned once.
        """
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from {RETRIEVAL_MODES}.")
        exclude = set(exclude_chunk_ids)
        pool = max(self.candidate_pool, top_k + len(exclude))

        if mode != "bm25_only":
            window_hits = self.vector_index.search(self.embedding_model.encode_query(query), top_k=pool * 4)
            dense_hits = _best_per_chunk(window_hits)[:pool]
            if mode == "dense_only":
                return self._ranked(dense_hits, top_k, exclude)

        bm25_hits = self.bm25_index.search(query, top_k=pool)
        if mode == "bm25_only":
            return self._ranked(bm25_hits, top_k, exclude)

        fused = reciprocal_rank_fusion(dense_hits, bm25_hits, k=RRF_K)
        if mode == "hybrid":
            return self._ranked(fused, top_k, exclude)

        candidates = self._ranked(fused, max(self.rerank_pool, top_k), exclude)
        return [chunk for chunk, _ in self.reranker.rerank(query, candidates, top_k=top_k)]


def load_processed_chunks(directory: Path) -> list[Chunk]:
    """Load chunks from ``chunks.parquet`` (pandas/pyarrow) or the dependency-free ``chunks.jsonl``."""
    directory = Path(directory)
    jsonl = directory / "chunks.jsonl"
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
    import pandas as pd

    return [Chunk.from_dict(row) for row in pd.read_parquet(directory / "chunks.parquet").to_dict("records")]


# ---------------------------------------------------------------------------
# 6. Retrieval Evaluation & Ablation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalEvalCase:
    """A question with known evidence.

    ``target_chunk_ids`` lists every chunk that counts as gold evidence. When
    ``evidence_groups`` is given, each group is one required piece of evidence that
    is found if *any* chunk in the group is retrieved (overlapping chunks often
    share the same evidence sentence). Without groups, each target id is its own group.
    """

    question: str
    target_chunk_ids: Sequence[str]
    evidence_groups: Sequence[Sequence[str]] = field(default_factory=tuple)
    qid: str = ""

    def groups(self) -> list[set[str]]:
        if self.evidence_groups:
            return [set(g) for g in self.evidence_groups]
        return [{cid} for cid in self.target_chunk_ids]


def recall_at_k(groups: Sequence[set[str]], retrieved_ids: Sequence[str], k: int) -> float:
    """Fraction of evidence groups with at least one chunk in the top ``k``."""
    if not groups:
        return 0.0
    top = set(retrieved_ids[:k])
    return sum(1 for g in groups if g & top) / len(groups)


def reciprocal_rank(groups: Sequence[set[str]], retrieved_ids: Sequence[str]) -> float:
    """1 / rank of the first retrieved chunk that belongs to any evidence group (0 if none)."""
    gold = set().union(*groups) if groups else set()
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def _normalize_quote(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_evidence_chunk_ids(chunks: Sequence[Chunk], document_name: str, quote: str, page: int | None = None) -> list[str]:
    """Ids of chunks from ``document_name`` (and ``page``, if given) whose text contains ``quote``."""
    needle = _normalize_quote(quote)
    return [
        c.chunk_id
        for c in chunks
        if c.document_name == document_name and (page is None or c.page == page) and needle in _normalize_quote(c.text)
    ]


def load_retrieval_eval_cases(path: Path, chunks: Sequence[Chunk]) -> list[RetrievalEvalCase]:
    """Read a retrieval eval JSONL file and resolve each evidence quote to gold chunk ids.

    Each ``evidence`` entry is one required piece of evidence: either a single
    ``{document_name, quote, page?}`` span or ``{"any_of": [span, ...]}`` when the same
    fact appears in several places. Unanswerable questions (no evidence) are skipped.
    A quote that matches no chunk raises ``ValueError`` so a stale or mistyped label
    can never silently score 0.
    """
    cases: list[RetrievalEvalCase] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("evidence"):
                continue
            groups = []
            for ev in row["evidence"]:
                ids: list[str] = []
                for alt in ev.get("any_of", [ev]):
                    found = resolve_evidence_chunk_ids(chunks, alt["document_name"], alt["quote"], alt.get("page"))
                    if not found:
                        raise ValueError(f"{row['qid']}: quote not found in {alt['document_name']} p.{alt.get('page')}: {alt['quote']!r}")
                    ids.extend(found)
                groups.append(tuple(dict.fromkeys(ids)))
            cases.append(
                RetrievalEvalCase(
                    question=row["question"],
                    target_chunk_ids=tuple(dict.fromkeys(cid for g in groups for cid in g)),
                    evidence_groups=tuple(groups),
                    qid=row["qid"],
                )
            )
    return cases


def evaluate_retrieval_ablation(
    eval_cases: Sequence[RetrievalEvalCase],
    retriever: HybridRetriever,
    output_csv: Path | None = None,
    per_query_csv: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Recall@5, Recall@10, MRR (over the top 10) and median latency for each retrieval mode."""
    results: dict[str, dict[str, float]] = {}
    per_query_rows: list[dict[str, object]] = []

    for mode in RETRIEVAL_MODES:
        r5, r10, rr, latencies = [], [], [], []
        for case in eval_cases:
            groups = case.groups()
            start = time.perf_counter()
            retrieved_ids = [c.chunk_id for c in retriever.retrieve(case.question, top_k=10, mode=mode)]
            latencies.append((time.perf_counter() - start) * 1000.0)
            r5.append(recall_at_k(groups, retrieved_ids, 5))
            r10.append(recall_at_k(groups, retrieved_ids, 10))
            rr.append(reciprocal_rank(groups, retrieved_ids))
            per_query_rows.append(
                {
                    "qid": case.qid,
                    "mode": mode,
                    "recall_at_5": round(r5[-1], 4),
                    "recall_at_10": round(r10[-1], 4),
                    "reciprocal_rank": round(rr[-1], 4),
                    "top_5_chunk_ids": " ".join(retrieved_ids[:5]),
                }
            )
        n = len(eval_cases)
        results[mode] = {
            "Recall@5": round(sum(r5) / n, 4) if n else 0.0,
            "Recall@10": round(sum(r10) / n, 4) if n else 0.0,
            "MRR": round(sum(rr) / n, 4) if n else 0.0,
            "n_queries": n,
            "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        }

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["configuration", "recall_at_5", "recall_at_10", "mrr", "n_queries", "median_latency_ms"])
            for mode in RETRIEVAL_MODES:
                m = results[mode]
                writer.writerow([MODE_LABELS[mode], m["Recall@5"], m["Recall@10"], m["MRR"], m["n_queries"], m["median_latency_ms"]])

    if per_query_csv:
        per_query_csv = Path(per_query_csv)
        per_query_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(per_query_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_query_rows[0]) if per_query_rows else ["qid"])
            writer.writeheader()
            writer.writerows(per_query_rows)

    return results
