"""Dashboard tests using Streamlit's AppTest — simulates real interaction
(typing, clicking) and inspects rendered output, no browser needed. The
dashboard's httpx calls hit a REAL uvicorn server (started in a background
thread, pointed at an isolated test index via the same dependency-override
pattern as tests/test_api.py) rather than a mocked HTTP client, so this is
genuine integration coverage of dashboard -> HTTP -> API -> index, not just
"did the script not crash."
"""
import threading
import time

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from src.api.app import app, get_index, get_raw_dir
from src.retrieval.index import HybridIndex
from tests.conftest import requires_groq

TEST_PORT = 8765
APP_PATH = "dashboard/app.py"


@pytest.fixture
def live_api(tmp_path, monkeypatch):
    index = HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_raw_dir] = lambda: tmp_path / "raw"

    config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("test API server did not start in time")

    monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{TEST_PORT}")

    yield index

    server.should_exit = True
    thread.join(timeout=5)
    app.dependency_overrides.clear()


def test_dashboard_loads_with_empty_index(live_api):
    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("No documents indexed yet." in c.value for c in at.caption)


def test_dashboard_shows_connection_error_when_api_unreachable(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")  # nothing listens here

    at = AppTest.from_file(APP_PATH)
    at.run()

    assert not at.exception
    assert any("Could not reach the API" in e.value for e in at.error)


def test_dashboard_low_confidence_question_renders_warning_without_groq(live_api, monkeypatch):
    # Same proof as the API-level test: this path never reaches Groq. A
    # genuinely irrelevant document is seeded first so this exercises the
    # "found candidates, none confident enough" message rather than the
    # simpler "no documents at all" one an empty index would trigger.
    from src.ingestion.models import Chunk

    live_api.add([Chunk(
        text="The company picnic is scheduled for next month at the downtown park.",
        source_path="notes.txt", chunk_index=0, strategy="structure",
    )])
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].input("What is the boiling point of mercury in Kelvin?")
    # Default AppTest timeout (3s) is too short here: this click triggers a
    # real HTTP round trip to the (real, in-process) API, which does real
    # embedding + reranking inference — including first-time model loading,
    # which is the slow part.
    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert len(at.warning) == 1
    assert "confidently relevant" in at.warning[0].value


@requires_groq
def test_dashboard_real_question_renders_answer_and_confidence(live_api):
    from src.ingestion.models import Chunk

    live_api.add([Chunk(
        text="Employees may work remotely up to 3 days per week.",
        source_path="policy.txt", chunk_index=0, strategy="structure",
    )])

    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].input("How many days can I work remotely?")
    # Generation + citation verification + confidence scoring are several
    # real LLM calls in sequence — needs real headroom, not the 3s default.
    at.button[0].click().run(timeout=60)

    assert not at.exception
    assert len(at.warning) == 0
    assert any("3" in m.value for m in at.markdown)
