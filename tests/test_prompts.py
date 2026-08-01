from src.generation.prompts import build_messages, format_context
from src.retrieval.reranker import RerankedHit


def _hit(chunk_id, text, source_path="doc.md", heading=None, page_number=-1):
    return RerankedHit(
        chunk_id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "heading": heading or "", "page_number": page_number},
        rerank_score=1.0,
        fused_score=1.0,
    )


def test_format_context_numbers_chunks_starting_at_one():
    chunks = [_hit("a", "First chunk text."), _hit("b", "Second chunk text.")]

    context = format_context(chunks)

    assert "[1]" in context
    assert "[2]" in context
    assert "First chunk text." in context
    assert "Second chunk text." in context
    assert context.index("[1]") < context.index("[2]")


def test_format_context_includes_heading_when_present():
    chunks = [_hit("a", "Some text.", source_path="guide.md", heading="Setup")]

    context = format_context(chunks)

    assert "guide.md" in context
    assert "Setup" in context


def test_format_context_omits_heading_when_absent():
    chunks = [_hit("a", "Some text.", source_path="notes.txt", heading=None)]

    context = format_context(chunks)

    assert "notes.txt" in context
    # no dangling separator when there's no heading to attach
    assert "notes.txt —" not in context


def test_format_context_includes_page_number_when_present():
    chunks = [_hit("a", "Some text.", source_path="manual.pdf", page_number=3)]

    context = format_context(chunks)

    assert "page 3" in context


def test_format_context_omits_page_number_sentinel():
    # page_number defaults to -1 (the "no page" sentinel from HybridIndex's
    # metadata convention) and should never leak into the prompt as text.
    chunks = [_hit("a", "Some text.", page_number=-1)]

    context = format_context(chunks)

    assert "page -1" not in context


def test_build_messages_includes_system_and_user_roles():
    chunks = [_hit("a", "Relevant text.")]

    messages = build_messages("What is the answer?", chunks)

    assert messages[0]["role"] == "system"
    assert "ONLY the provided context" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "What is the answer?" in messages[1]["content"]
    assert "Relevant text." in messages[1]["content"]
