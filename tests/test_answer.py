from src.ingestion.models import Chunk
from src.generation.answer import LOW_CONFIDENCE_THRESHOLD, _build_low_confidence_answer, answer_question
from src.retrieval.index import HybridIndex
from src.retrieval.reranker import RerankedHit
from tests.conftest import requires_groq


def _hit(text, source_path="doc.md", heading=None):
    return RerankedHit(
        chunk_id="x", text=text, metadata={"source_path": source_path, "heading": heading or ""},
        rerank_score=1.0, fused_score=1.0,
    )


def _index(tmp_path):
    return HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")


def _chunk(text, index=0, source="notes.md"):
    return Chunk(text=text, source_path=source, chunk_index=index, strategy="structure")


# --- _build_low_confidence_answer: pure formatting logic, no model needed ---

def test_low_confidence_answer_lists_candidate_sources():
    text = _build_low_confidence_answer([_hit("some text", source_path="policy.md", heading="Remote Work")])
    assert "policy.md" in text
    assert "Remote Work" in text


def test_low_confidence_answer_handles_missing_heading():
    text = _build_low_confidence_answer([_hit("some text", source_path="notes.txt", heading=None)])
    assert "notes.txt" in text


# --- answer_question: empty index short-circuits with no chunks at all ---

def test_answer_question_empty_index_is_low_confidence(tmp_path):
    index = _index(tmp_path)
    result = answer_question(index, "Anything?")
    assert result.low_confidence is True
    assert result.sources == []


# --- answer_question: low retrieval confidence short-circuits BEFORE any
# LLM call — proven here by removing GROQ_API_KEY entirely and confirming
# the call still succeeds, since a genuinely irrelevant question should
# never reach the point where a Groq client is even constructed.

def test_answer_question_low_confidence_never_calls_the_llm(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    index = _index(tmp_path)
    index.add([_chunk("The company picnic is scheduled for next month at the downtown park.")])

    result = answer_question(index, "What is the boiling point of mercury in Kelvin?")

    assert result.low_confidence is True
    assert "notes.md" in result.text
    assert result.confidence.retrieval_confidence < LOW_CONFIDENCE_THRESHOLD
    assert result.confidence.citation_coverage == 0.0
    assert result.confidence.completeness == 0.0


def test_min_confidence_is_configurable(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    index = _index(tmp_path)
    index.add([_chunk("The company picnic is scheduled for next month at the downtown park.")])

    # threshold of 0.0 means nothing is ever "too low" -- but the client
    # still needs to exist to actually generate, so this should now raise
    # trying to reach Groq rather than gracefully short-circuiting.
    import pytest
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        answer_question(index, "What is the boiling point of mercury in Kelvin?", min_confidence=0.0)


# --- answer_question: real generation path, needs a real Groq call ---

@requires_groq
def test_answer_question_relevant_question_generates_a_real_answer(tmp_path):
    index = _index(tmp_path)
    index.add([_chunk("Employees may work remotely up to 3 days per week.", source="policy.md")])

    result = answer_question(index, "How many days can I work remotely?")

    assert result.low_confidence is False
    assert "3" in result.text
    assert result.confidence.retrieval_confidence >= LOW_CONFIDENCE_THRESHOLD
