"""Streamlit dashboard for the Hybrid RAG Document Assistant.

A thin HTTP client over the FastAPI service (src/api/app.py) — it calls
/v1/ask, /v1/documents, /v1/ingest the same way any other client would,
rather than importing the pipeline directly. This matches how the two
pieces will actually be deployed later (separate API and frontend
services in Docker Compose), and means this file only has to get the UI
right — the underlying behavior is already tested at the API layer.

Run (two terminals):
    venv\\Scripts\\python.exe -m uvicorn src.api.app:app
    venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""
import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
SUPPORTED_TYPES = ["txt", "md", "html", "htm", "pdf"]
STRATEGIES = ["structure", "fixed", "semantic"]


def api_get(path: str):
    response = httpx.get(f"{API_BASE_URL}{path}", timeout=30.0)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs):
    response = httpx.post(f"{API_BASE_URL}{path}", timeout=90.0, **kwargs)
    response.raise_for_status()
    return response.json()


def render_documents_sidebar():
    st.header("Documents")

    uploaded_files = st.file_uploader("Upload documents", type=SUPPORTED_TYPES, accept_multiple_files=True)
    strategy = st.selectbox("Chunking strategy", STRATEGIES)

    if uploaded_files and st.button("Ingest"):
        files = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        try:
            result = api_post("/v1/ingest", data={"strategy": strategy}, files=files)
        except httpx.ConnectError:
            st.error("Could not reach the API — is it running? (uvicorn src.api.app:app)")
        except httpx.HTTPStatusError as exc:
            st.error(f"Ingest failed: {exc.response.text}")
        else:
            for ingested in result["files"]:
                st.success(f"{ingested['filename']}: {ingested['chunks_added']} added, {ingested['chunks_skipped']} skipped")

    st.divider()

    try:
        docs = api_get("/v1/documents")
    except httpx.ConnectError:
        st.error("Could not reach the API — is it running?")
        return

    st.caption(f"{docs['chunk_count']} chunk(s) indexed")
    if docs["documents"]:
        for name in docs["documents"]:
            st.text(f"\U0001F4C4 {name}")
    else:
        st.caption("No documents indexed yet.")


def render_confidence(confidence: dict):
    st.subheader("Confidence")
    columns = st.columns(4)
    columns[0].metric("Retrieval", f"{confidence['retrieval_confidence']:.2f}")
    columns[1].metric("Citation coverage", f"{confidence['citation_coverage']:.2f}")
    columns[2].metric("Completeness", f"{confidence['completeness']:.2f}")
    columns[3].metric("Composite", f"{confidence['composite']:.2f}")


def render_citations(citations: list):
    st.subheader("Citations")
    for citation in citations:
        icon = "✅" if citation["supported"] else "❌"
        with st.expander(f"{icon} [{citation['citation_number']}] {citation['claim'][:80]}"):
            st.write(citation["claim"])
            st.caption(citation["reasoning"])


def render_sources(sources: list):
    st.subheader("Retrieved sources")
    for i, source in enumerate(sources, start=1):
        heading = source["metadata"].get("heading")
        source_path = source["metadata"].get("source_path", "unknown")
        label = f"[{i}] {source_path}" + (f" — {heading}" if heading else "")
        with st.expander(f"{label} (rerank={source['rerank_score']:.2f})"):
            st.write(source["text"])


def render_ask_panel():
    question = st.text_input("Ask a question about your documents")

    if not (st.button("Ask") and question):
        return

    try:
        with st.spinner("Thinking..."):
            answer = api_post("/v1/ask", json={"question": question})
    except httpx.ConnectError:
        st.error("Could not reach the API — is it running? (uvicorn src.api.app:app)")
        return
    except httpx.HTTPStatusError as exc:
        st.error(f"Request failed: {exc.response.text}")
        return

    if answer["low_confidence"]:
        st.warning(answer["text"])
    else:
        st.markdown(answer["text"])

    if answer["confidence"]:
        render_confidence(answer["confidence"])
    if answer["citations"]:
        render_citations(answer["citations"])
    if answer["sources"]:
        render_sources(answer["sources"])


st.set_page_config(page_title="Hybrid RAG Document Assistant", layout="wide")
st.title("Hybrid RAG Document Assistant")

with st.sidebar:
    render_documents_sidebar()

render_ask_panel()
