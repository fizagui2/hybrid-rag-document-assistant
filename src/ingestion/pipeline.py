"""Top-level ingestion entrypoints: load a file (or a whole directory) and
persist the normalized result to data/processed/.
"""
from pathlib import Path

from .loaders import SUPPORTED_EXTENSIONS, load_document
from .models import Document
from .store import save_processed


def ingest_file(path: str | Path) -> Document:
    document = load_document(path)
    save_processed(document)
    return document


def ingest_directory(directory: str | Path = "data/raw") -> list[Document]:
    directory = Path(directory)
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [ingest_file(path) for path in files]
