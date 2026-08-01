from pathlib import Path

import pytest

from src.ingestion.chunker import _group_by_breakpoints, chunk_document
from src.ingestion.loaders import load_document
from src.ingestion.models import Document, Section

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixed_chunking_respects_chunk_size():
    doc = load_document(FIXTURES / "sample.md")
    chunks = chunk_document(doc, strategy="fixed", chunk_size=20, overlap=5)
    assert len(chunks) > 1
    assert all(c.strategy == "fixed" for c in chunks)
    assert all(len(c.text) <= 20 for c in chunks)


def test_fixed_chunking_rejects_overlap_gte_chunk_size():
    doc = load_document(FIXTURES / "sample.txt")
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="fixed", chunk_size=10, overlap=10)


def test_structure_chunking_keeps_short_sections_whole():
    doc = load_document(FIXTURES / "sample.md")
    chunks = chunk_document(doc, strategy="structure", chunk_size=500, overlap=50)
    # both sections in sample.md are short, so one chunk per section
    assert len(chunks) == 2
    assert chunks[0].heading == "Introduction"
    assert chunks[1].heading == "Details"
    assert all(c.strategy == "structure" for c in chunks)


def test_structure_chunking_splits_long_sections():
    doc = load_document(FIXTURES / "sample.md")
    chunks = chunk_document(doc, strategy="structure", chunk_size=15, overlap=0)
    assert len(chunks) > 2  # sections had to be split further
    assert all(len(c.text) <= 15 for c in chunks)
    assert all(c.heading in ("Introduction", "Details") for c in chunks)


def test_chunk_index_is_sequential():
    doc = load_document(FIXTURES / "sample.md")
    chunks = chunk_document(doc, strategy="structure", chunk_size=15, overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_unknown_strategy_raises():
    doc = load_document(FIXTURES / "sample.txt")
    with pytest.raises(ValueError):
        chunk_document(doc, strategy="not_a_real_strategy")


def test_char_count_matches_text_length():
    doc = load_document(FIXTURES / "sample.txt")
    chunks = chunk_document(doc, strategy="fixed")
    assert chunks[0].char_count == len(chunks[0].text)


# --- semantic chunking ---
# _group_by_breakpoints is the core decision logic (where to cut) and is
# pure Python — no embedding model needed, so it's tested directly with
# synthetic distances for a fast, deterministic check of the algorithm.

def test_group_by_breakpoints_splits_at_high_distance():
    sentences = ["A.", "B.", "C.", "D."]
    distances = [0.1, 0.9, 0.1]  # big jump only between sentence B and C
    segments = _group_by_breakpoints(sentences, distances, threshold=0.5)
    assert segments == ["A. B.", "C. D."]


def test_group_by_breakpoints_no_breaks_below_threshold():
    sentences = ["A.", "B.", "C."]
    distances = [0.2, 0.3]
    segments = _group_by_breakpoints(sentences, distances, threshold=0.5)
    assert segments == ["A. B. C."]


# The rest exercise chunk_semantic end to end against the real embedding
# model (downloaded once, then cached locally by Hugging Face).

def test_semantic_chunking_separates_distinct_topics():
    text = (
        "Cats are popular pets known for their independence. "
        "Many cat owners enjoy their playful and curious nature. "
        "Photosynthesis converts sunlight into chemical energy in plants. "
        "Chlorophyll absorbs light primarily in the blue and red wavelengths."
    )
    doc = Document(source_path="demo.md", format="md", sections=[Section(text=text, heading="Mixed Topics")])
    chunks = chunk_document(doc, strategy="semantic", chunk_size=1000, overlap=0)
    # should NOT merge cats + photosynthesis into one chunk just because it fits
    assert len(chunks) >= 2
    assert all(c.strategy == "semantic" for c in chunks)
    assert all(c.heading == "Mixed Topics" for c in chunks)


def test_semantic_chunking_respects_chunk_size():
    doc = load_document(FIXTURES / "sample.md")
    chunks = chunk_document(doc, strategy="semantic", chunk_size=20, overlap=0)
    assert all(len(c.text) <= 20 for c in chunks)


def test_semantic_chunking_single_sentence_section():
    doc = load_document(FIXTURES / "sample.txt")
    chunks = chunk_document(doc, strategy="semantic")
    assert len(chunks) == 1
    assert chunks[0].strategy == "semantic"
