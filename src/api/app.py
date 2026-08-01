"""FastAPI service exposing the RAG pipeline: ingest documents, list what's
indexed, and ask grounded questions. Only /v1/ask touches Groq — the other
two endpoints are entirely local.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request, UploadFile

from ..generation.answer import answer_question
from ..ingestion.chunker import chunk_document
from ..ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from ..retrieval.index import HybridIndex
from .schemas import AskRequest, AskResponse, DocumentsResponse, IngestedFile, IngestResponse

RAW_DIR = Path("data/raw")
DEFAULT_INGEST_STRATEGY = "structure"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One HybridIndex for the app's lifetime, pointed at the real
    # persistent data/chroma + data/bm25_index.pkl — not a fresh index per
    # request, and not a throwaway/test one.
    app.state.index = HybridIndex()
    app.state.raw_dir = RAW_DIR
    yield


app = FastAPI(title="Hybrid RAG Document Assistant", lifespan=lifespan)


def get_index(request: Request) -> HybridIndex:
    return request.app.state.index


def get_raw_dir(request: Request) -> Path:
    # A dependency, not the RAW_DIR module constant used directly — so
    # tests can override it (app.dependency_overrides) the same way they
    # override get_index, instead of every /v1/ingest test silently
    # writing real files into the project's actual data/raw/.
    return request.app.state.raw_dir


@app.get("/v1/documents", response_model=DocumentsResponse)
def list_documents(index: HybridIndex = Depends(get_index)) -> DocumentsResponse:
    return DocumentsResponse(documents=index.list_documents(), chunk_count=index.count())


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile],
    # Form(...), not a plain default — alongside UploadFile params (which
    # force multipart/form-data), a bare `str = default` parameter is
    # treated as a query parameter, not a form field, so a client actually
    # submitting `strategy` in the form body would be silently ignored and
    # the default used instead. Caught by a test asserting an invalid
    # strategy gets rejected — it didn't, because it was never being read.
    strategy: str = Form(DEFAULT_INGEST_STRATEGY),
    index: HybridIndex = Depends(get_index),
    raw_dir: Path = Depends(get_raw_dir),
) -> IngestResponse:
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {upload.filename!r}")

        destination = raw_dir / upload.filename
        destination.write_bytes(await upload.read())

        document = load_document(destination)
        try:
            chunks = chunk_document(document, strategy=strategy)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = index.add(chunks)
        results.append(
            IngestedFile(filename=upload.filename, chunks_added=result.added, chunks_skipped=len(result.skipped))
        )

    return IngestResponse(strategy=strategy, files=results)


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest, index: HybridIndex = Depends(get_index)) -> AskResponse:
    answer = answer_question(index, request.question, final_k=request.final_k)
    return AskResponse.model_validate(answer)
