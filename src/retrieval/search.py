"""Hybrid search: run dense + sparse retrieval and fuse the two ranked lists
with Reciprocal Rank Fusion (RRF).

RRF combines results by *rank position*, not raw score, specifically because
dense cosine similarities (~0-1) and BM25 scores (unbounded, can even be
negative on tiny corpora) live on incomparable scales — there's no sane way
to add them directly. weight / (rrf_k + rank) sidesteps that: a chunk ranked
#1 in a list contributes far more than one ranked #20, regardless of what
scoring system produced that ranking.
"""
from dataclasses import dataclass
from typing import Optional

from .index import HybridIndex, SearchHit

DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 0.7
DEFAULT_SPARSE_WEIGHT = 0.3
DEFAULT_CANDIDATE_K = 20


@dataclass
class FusedHit:
    chunk_id: str
    text: str
    metadata: dict
    fused_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None


def reciprocal_rank_fusion(
    dense_hits: list[SearchHit],
    sparse_hits: list[SearchHit],
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Merge two ranked SearchHit lists into one ranked FusedHit list."""
    scores: dict[str, float] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}
    lookup: dict[str, SearchHit] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + dense_weight / (rrf_k + rank)
        dense_ranks[hit.chunk_id] = rank
        lookup[hit.chunk_id] = hit

    for rank, hit in enumerate(sparse_hits, start=1):
        scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + sparse_weight / (rrf_k + rank)
        sparse_ranks[hit.chunk_id] = rank
        lookup.setdefault(hit.chunk_id, hit)

    fused = [
        FusedHit(
            chunk_id=chunk_id,
            text=lookup[chunk_id].text,
            metadata=lookup[chunk_id].metadata,
            fused_score=score,
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda hit: hit.fused_score, reverse=True)
    return fused


def hybrid_search(
    index: HybridIndex,
    query: str,
    k: int = 10,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Retrieve candidate_k results from each method (more than the final k,
    so fusion has enough candidates to reorder from — not just each method's
    independent top-k), fuse them, and return the top k.
    """
    dense_hits = index.search_dense(query, k=candidate_k)
    sparse_hits = index.search_sparse(query, k=candidate_k)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, dense_weight, sparse_weight, rrf_k)
    return fused[:k]
