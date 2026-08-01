"""Cross-encoder reranking: a second, more accurate but far more expensive
pass over a small shortlist of candidates from hybrid search.

Dense retrieval uses a *bi-encoder* (src/embeddings.py): query and passage
are embedded separately, so passage embeddings can be precomputed once and
compared to a query via a single cheap vector operation — that's what makes
searching a whole corpus fast. A *cross-encoder* is the opposite: it takes
the query and one candidate together as a single input, runs them jointly
through the model so query and passage tokens can directly attend to each
other, and outputs one relevance score per pair. That joint attention makes
it meaningfully more accurate at judging "does this passage really answer
this query" — but nothing can be precomputed, so it can't scale to an entire
corpus. It's only affordable as a reranking step over a short candidate list
hybrid search already narrowed down.

retrieve() ties the whole Phase 2 pipeline together: hybrid_search() narrows
a whole corpus down to candidate_k fused candidates, then rerank() reorders
those candidate_k down to the final_k actually returned.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from sentence_transformers import CrossEncoder

from .index import HybridIndex
from .search import DEFAULT_CANDIDATE_K, DEFAULT_DENSE_WEIGHT, DEFAULT_SPARSE_WEIGHT, FusedHit, hybrid_search

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_FINAL_K = 5


@dataclass
class RerankedHit:
    chunk_id: str
    text: str
    metadata: dict
    rerank_score: float
    fused_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder model, same lazy-singleton pattern
    as get_embedder() in src/embeddings.py.
    """
    return CrossEncoder(MODEL_NAME)


def rerank(query: str, candidates: list[FusedHit], top_k: int = DEFAULT_FINAL_K) -> list[RerankedHit]:
    """Score every candidate jointly with the query and return the top_k,
    reordered by that (far more expensive, far more accurate) score.
    """
    if not candidates:
        return []

    pairs = [(query, hit.text) for hit in candidates]
    scores = get_reranker().predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RerankedHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=hit.metadata,
            rerank_score=float(score),
            fused_score=hit.fused_score,
            dense_rank=hit.dense_rank,
            sparse_rank=hit.sparse_rank,
        )
        for hit, score in ranked[:top_k]
    ]


def retrieve(
    index: HybridIndex,
    query: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> list[RerankedHit]:
    """The full Phase 2 pipeline: hybrid search down to candidate_k
    candidates, cross-encoder rerank down to the final_k actually returned.
    """
    candidates = hybrid_search(
        index, query, k=candidate_k, candidate_k=candidate_k, dense_weight=dense_weight, sparse_weight=sparse_weight
    )
    return rerank(query, candidates, top_k=final_k)
