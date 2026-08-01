from src.ingestion.models import Chunk
from src.retrieval.index import HybridIndex, SearchHit
from src.retrieval.search import hybrid_search, reciprocal_rank_fusion


def _hit(chunk_id, text="text", score=1.0):
    return SearchHit(chunk_id=chunk_id, text=text, metadata={}, score=score)


def _make_chunk(text, index, source="doc.md", strategy="structure", heading=None):
    return Chunk(text=text, source_path=source, chunk_index=index, strategy=strategy, heading=heading)


def _make_index(tmp_path):
    return HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")


# --- reciprocal_rank_fusion: pure logic, no model needed ---

def test_rrf_ranks_item_present_in_both_lists_first():
    dense = [_hit("a"), _hit("b"), _hit("c")]
    sparse = [_hit("b"), _hit("c"), _hit("a")]

    fused = reciprocal_rank_fusion(dense, sparse)

    # "b" is #2 dense / #1 sparse, "a" is #1 dense / #3 sparse — with default
    # weights close to even, both outrank "c" (#3 dense / #2 sparse), and
    # the top spot goes to whichever combination scores higher.
    assert fused[0].chunk_id in ("a", "b")
    assert {hit.chunk_id for hit in fused} == {"a", "b", "c"}


def test_rrf_includes_items_present_in_only_one_list():
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("c")]

    fused = reciprocal_rank_fusion(dense, sparse)

    assert {hit.chunk_id for hit in fused} == {"a", "b", "c"}
    fused_by_id = {hit.chunk_id: hit for hit in fused}
    assert fused_by_id["c"].dense_rank is None
    assert fused_by_id["c"].sparse_rank == 1
    assert fused_by_id["a"].sparse_rank is None


def test_rrf_weight_shifts_ranking_toward_that_method():
    # "a" appears only in dense (rank 1); "b" appears only in sparse (rank
    # 1) — each list has exactly one item, so the comparison isolates the
    # weight's effect instead of also being influenced by "appears in both
    # lists" (which, as test_rrf_ranks_item_present_in_both_lists_first
    # shows, can itself outscore a single top rank — a different effect).
    dense = [_hit("a")]
    sparse = [_hit("b")]

    dense_favored = reciprocal_rank_fusion(dense, sparse, dense_weight=0.9, sparse_weight=0.1)
    sparse_favored = reciprocal_rank_fusion(dense, sparse, dense_weight=0.1, sparse_weight=0.9)

    assert dense_favored[0].chunk_id == "a"
    assert sparse_favored[0].chunk_id == "b"


def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([], []) == []


# --- search_dense / search_sparse / hybrid_search: real models ---

def test_search_dense_ranks_semantically_closest_chunk_first(tmp_path):
    index = _make_index(tmp_path)
    index.add([
        _make_chunk("The cat sat on the mat.", 0),
        _make_chunk("Quarterly revenue increased significantly.", 1),
    ])

    hits = index.search_dense("a feline resting on a rug", k=2)

    assert hits[0].text == "The cat sat on the mat."
    assert hits[0].score > hits[1].score


def test_search_sparse_finds_exact_term_match(tmp_path):
    index = _make_index(tmp_path)
    # 3 chunks, not 2: with exactly 2 documents and a term in exactly 1 of
    # them, BM25's idf works out to log((2-1+0.5)/(1+0.5)) == log(1) == 0
    # exactly — a real degenerate case at that specific corpus size, not a
    # bug (confirmed by reproducing it directly against rank_bm25). A third,
    # unrelated document avoids landing on that exact edge case.
    index.add([
        _make_chunk("The system returned error code ERR_502 during startup.", 0),
        _make_chunk("Configuration files are loaded from the config directory.", 1),
        _make_chunk("The onboarding guide covers account setup steps.", 2),
    ])

    hits = index.search_sparse("ERR_502", k=5)

    assert len(hits) == 1
    assert "ERR_502" in hits[0].text


def test_search_sparse_drops_non_matching_chunks(tmp_path):
    index = _make_index(tmp_path)
    index.add([_make_chunk("Completely unrelated content about gardening.", 0)])

    hits = index.search_sparse("ERR_502", k=5)

    assert hits == []


def test_search_dense_and_sparse_empty_index(tmp_path):
    index = _make_index(tmp_path)
    assert index.search_dense("anything") == []
    assert index.search_sparse("anything") == []


def test_hybrid_search_surfaces_both_keyword_and_semantic_matches(tmp_path):
    index = _make_index(tmp_path)
    index.add([
        _make_chunk("The system returned error code ERR_502 during startup.", 0),
        _make_chunk("A malfunction occurred while the service was initializing.", 1),
        _make_chunk("Our reranker is a cross-encoder scored against the query.", 2),
    ])

    # ERR_502 is an exact term only chunk 0 has; chunk 1 is a semantic
    # paraphrase of "startup failure" with no shared vocabulary at all.
    # Hybrid search should surface both ahead of the unrelated chunk 2.
    results = hybrid_search(index, "startup failure ERR_502", k=3)

    top_two_ids = {hit.chunk_id for hit in results[:2]}
    assert HybridIndex.chunk_id(_make_chunk("x", 0)) in top_two_ids
    assert HybridIndex.chunk_id(_make_chunk("x", 1)) in top_two_ids
