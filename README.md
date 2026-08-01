# Hybrid RAG Document Assistant

## About this project

This project was built as a student exercise to test how reliable AI (Claude Code) is in assisting with building a real, working piece of software end-to-end — not just write code, but actually run it, catch real bugs, and fix them.

A lot of genuine issues came up along the way and got documented as they happened: sentence-splitting logic silently detaching citations from claims, a chunking bug that averaged relevance scores instead of taking the max, Groq API rate limits hit multiple times (including a real lesson about not trusting a single successful test call as proof of quota availability), a `.bat` script command that failed under certain shells, and more.

## Overview

A Retrieval-Augmented Generation (RAG) system: upload your own documents, and ask questions about them. It combines semantic search and keyword search, reranks results for accuracy, generates grounded answers with citations, verifies those citations, and scores its own confidence while refusing to answer rather than guessing when it isn't confident enough.

## How to run it

**Easiest**: double-click `run.bat` (requires Docker Desktop to be installed and running). It starts the app and opens it in your browser automatically.

**Manual (Docker)**:
```
docker compose up -d
```
Then visit `http://localhost:8501`. Stop with `docker compose down` (or double-click `stop.bat`).

**Local (no Docker)**, two terminals:
```
venv\Scripts\python.exe -m uvicorn src.api.app:app
venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

See `DOCKER.md` for more Docker commands (logs, resetting data, etc.)

## Tech stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **Vector search**: ChromaDB (local, embedded)
- **Keyword search**: BM25 (`rank_bm25`)
- **Embeddings + reranking**: local `sentence-transformers` models (free, no API key)
- **Answer generation**: Groq API (free tier)
- **Containerization**: Docker + Docker Compose
