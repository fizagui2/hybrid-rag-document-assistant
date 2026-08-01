from src.ingestion.models import Chunk
from src.retrieval.index import HybridIndex


def _make_chunk(text, index=0, source="doc.md", strategy="structure", heading=None):
    return Chunk(text=text, source_path=source, chunk_index=index, strategy=strategy, heading=heading)


def _make_index(tmp_path):
    return HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")


def test_add_populates_both_indexes(tmp_path):
    index = _make_index(tmp_path)
    chunks = [_make_chunk("The cat sat on the mat.", 0), _make_chunk("Dogs bark loudly.", 1)]

    result = index.add(chunks)

    assert result.added == 2
    assert result.skipped == []
    assert index.count() == 2
    assert len(index.bm25_ids) == 2


def test_reindexing_same_chunk_does_not_duplicate(tmp_path):
    index = _make_index(tmp_path)
    index.add([_make_chunk("Original text.", 0)])
    result = index.add([_make_chunk("Original text.", 0)])  # same id: same source/strategy/index

    assert index.count() == 1
    assert len(index.bm25_ids) == 1
    # This must go through the normal upsert path, not get treated as a
    # near-duplicate of itself and skipped — those are very different things.
    assert result.added == 1
    assert result.skipped == []


def test_reindexing_same_id_updates_bm25_text(tmp_path):
    index = _make_index(tmp_path)
    chunk_id = HybridIndex.chunk_id(_make_chunk("apples and oranges", 0))
    index.add([_make_chunk("apples and oranges", 0)])
    index.add([_make_chunk("bananas and grapes", 0)])  # same id, different text

    assert index.count() == 1
    assert len(index.bm25_ids) == 1
    # The tokenized corpus for that id should reflect the *new* text, not
    # the stale old one. (Not testing via get_scores() here — BM25's IDF
    # term goes negative for any term when the corpus has exactly one
    # document, since a term in 100% of documents is treated as
    # uninformative. That's a real property of BM25 on tiny corpora, not
    # something to work around by asserting on it.)
    assert index._bm25_tokenized_by_id[chunk_id] == ["bananas", "and", "grapes"]


def test_bm25_persists_across_instances(tmp_path):
    chroma_dir = tmp_path / "chroma"
    bm25_path = tmp_path / "bm25.pkl"

    first = HybridIndex(chroma_dir=chroma_dir, bm25_path=bm25_path)
    first.add([_make_chunk("Persisted content here.", 0)])

    second = HybridIndex(chroma_dir=chroma_dir, bm25_path=bm25_path)
    assert len(second.bm25_ids) == 1
    assert second.count() == 1


def test_metadata_handles_none_heading_and_page(tmp_path):
    index = _make_index(tmp_path)
    chunk = Chunk(text="No heading here.", source_path="doc.txt", chunk_index=0, strategy="fixed")

    index.add([chunk])

    result = index._collection.get(ids=[index.chunk_id(chunk)], include=["metadatas"])
    metadata = result["metadatas"][0]
    assert metadata["heading"] == ""
    assert metadata["page_number"] == -1


def test_empty_add_is_a_no_op(tmp_path):
    index = _make_index(tmp_path)
    result = index.add([])
    assert result.added == 0
    assert result.skipped == []
    assert index.count() == 0


# --- near-duplicate detection ---
# These exercise the real embedding model, since "is this a near-duplicate"
# is a genuine semantic question a hardcoded fixture can't answer reliably.

def test_near_duplicate_is_skipped(tmp_path):
    index = _make_index(tmp_path)
    index.add([_make_chunk("The quick brown fox jumps over the lazy dog.", 0)])

    # one-word paraphrase of the same sentence — should read as a near-duplicate
    result = index.add([_make_chunk("The quick brown fox jumps over a lazy dog.", 1)])

    assert result.added == 0
    assert len(result.skipped) == 1
    assert result.skipped[0].similarity > 0.95
    assert index.count() == 1  # the near-duplicate was never inserted


def test_dissimilar_chunks_are_both_added(tmp_path):
    index = _make_index(tmp_path)
    index.add([_make_chunk("The quick brown fox jumps over the lazy dog.", 0)])

    result = index.add([_make_chunk("Quarterly revenue increased by twelve percent.", 1)])

    assert result.added == 1
    assert result.skipped == []
    assert index.count() == 2


def test_within_batch_duplicates_are_caught(tmp_path):
    index = _make_index(tmp_path)

    result = index.add([
        _make_chunk("The quick brown fox jumps over the lazy dog.", 0),
        _make_chunk("The quick brown fox jumps over a lazy dog.", 1),  # near-dup of chunk 0
        _make_chunk("Quarterly revenue increased by twelve percent.", 2),
    ])

    assert result.added == 2
    assert len(result.skipped) == 1
    assert result.skipped[0].duplicate_of == HybridIndex.chunk_id(_make_chunk("x", 0))
    assert index.count() == 2


def test_dedupe_threshold_is_configurable(tmp_path):
    index = _make_index(tmp_path)
    index.add([_make_chunk("The quick brown fox jumps over the lazy dog.", 0)])

    # same paraphrase as the near-duplicate test, but with an unreachably
    # strict threshold — should NOT be flagged this time
    result = index.add(
        [_make_chunk("The quick brown fox jumps over a lazy dog.", 1)],
        dedupe_threshold=0.999999,
    )

    assert result.added == 1
    assert result.skipped == []
