import io

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, get_index, get_raw_dir
from src.retrieval.index import HybridIndex
from tests.conftest import requires_groq


@pytest.fixture
def client(tmp_path):
    index = HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_raw_dir] = lambda: tmp_path / "raw"
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _upload_txt(client, filename, content):
    return client.post(
        "/v1/ingest", files=[("files", (filename, io.BytesIO(content.encode("utf-8")), "text/plain"))]
    )


# --- /v1/documents ---

def test_documents_empty_index(client):
    response = client.get("/v1/documents")
    assert response.status_code == 200
    assert response.json() == {"documents": [], "chunk_count": 0}


def test_documents_reflects_ingested_files(client):
    _upload_txt(client, "notes.txt", "The company picnic is scheduled for next month.")

    response = client.get("/v1/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == ["notes.txt"]
    assert body["chunk_count"] > 0


# --- /v1/ingest ---

def test_ingest_returns_chunk_counts(client):
    response = _upload_txt(client, "policy.txt", "Employees may work remotely up to 3 days per week.")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "structure"
    assert body["files"][0]["filename"] == "policy.txt"
    assert body["files"][0]["chunks_added"] == 1
    assert body["files"][0]["chunks_skipped"] == 0


def test_ingest_rejects_unsupported_file_type(client):
    response = client.post(
        "/v1/ingest", files=[("files", ("data.xyz", io.BytesIO(b"whatever"), "application/octet-stream"))]
    )
    assert response.status_code == 400


def test_ingest_rejects_invalid_strategy(client):
    response = client.post(
        "/v1/ingest",
        data={"strategy": "not_a_real_strategy"},
        files=[("files", ("notes.txt", io.BytesIO(b"Some content here."), "text/plain"))],
    )
    assert response.status_code == 400


def test_ingest_reindexing_same_file_does_not_duplicate(client):
    _upload_txt(client, "notes.txt", "Some fixed content.")
    _upload_txt(client, "notes.txt", "Some fixed content.")

    response = client.get("/v1/documents")

    assert response.json()["chunk_count"] == 1


# --- /v1/ask ---

def test_ask_rejects_empty_question(client):
    response = client.post("/v1/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_empty_index_declines_without_calling_groq(client, monkeypatch):
    # Proves this path never reaches Groq: removing the key entirely and
    # confirming the request still succeeds.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    response = client.post("/v1/ask", json={"question": "Anything at all?"})

    assert response.status_code == 200
    body = response.json()
    assert body["low_confidence"] is True
    assert body["sources"] == []


@requires_groq
def test_ask_returns_a_real_grounded_answer(client):
    _upload_txt(client, "policy.txt", "Employees may work remotely up to 3 days per week.")

    response = client.post("/v1/ask", json={"question": "How many days can I work remotely?"})

    assert response.status_code == 200
    body = response.json()
    assert body["low_confidence"] is False
    assert "3" in body["text"]
