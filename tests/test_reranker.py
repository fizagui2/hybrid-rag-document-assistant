from src.ingestion.models import Chunk
from src.retrieval.index import HybridIndex
from src.retrieval.reranker import RerankedHit, rerank, retrieve
from src.retrieval.search import FusedHit


def _make_chunk(text, index, source="doc.md", strategy="structure", heading=None):
    return Chunk(text=text, source_path=source, chunk_index=index, strategy=strategy, heading=heading)


def _make_index(tmp_path):
    return HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")


def _fused(chunk_id, text, fused_score=0.0):
    return FusedHit(chunk_id=chunk_id, text=text, metadata={}, fused_score=fused_score)


def test_rerank_corrects_a_misleading_fusion_order():
    # "a" is deliberately given the higher fused_score even though it only
    # superficially relates to the query; "b" directly answers it. A real
    # cross-encoder, unlike the bi-encoder/BM25 combination that produced
    # these fused_scores, should recognize "b" as the true answer.
    candidates = [
        _fused("a", "France is a country in Europe known for its cuisine and wine.", fused_score=0.9),
        _fused("b", "Paris is the capital of France.", fused_score=0.1),
    ]

    reranked = rerank("What is the capital of France?", candidates, top_k=2)

    assert reranked[0].chunk_id == "b"
    assert reranked[0].rerank_score > reranked[1].rerank_score
    # fused_score is carried through unchanged for reference, not overwritten
    assert reranked[0].fused_score == 0.1


def test_rerank_respects_top_k():
    candidates = [_fused(str(i), f"Chunk number {i} about various topics.") for i in range(10)]

    reranked = rerank("various topics", candidates, top_k=3)

    assert len(reranked) == 3
    assert all(isinstance(hit, RerankedHit) for hit in reranked)


def test_rerank_empty_candidates():
    assert rerank("anything", []) == []


def test_retrieve_full_pipeline(tmp_path):
    index = _make_index(tmp_path)
    index.add([
        _make_chunk("Paris is the capital of France.", 0),
        _make_chunk("The Eiffel Tower is a famous landmark in Paris.", 1),
        _make_chunk("Quarterly revenue increased significantly this year.", 2),
    ])

    results = retrieve(index, "What is the capital of France?", candidate_k=10, final_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == HybridIndex.chunk_id(_make_chunk("x", 0))
