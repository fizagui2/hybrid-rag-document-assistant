"""Seed script: ingest the eval corpus (eval/corpus/) into the index so
anyone spinning up the app (locally or via Docker Compose) sees real,
queryable data immediately instead of an empty index.

Usage:
    venv\\Scripts\\python.exe scripts\\seed.py
    docker compose run api python scripts/seed.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_document  # noqa: E402
from src.ingestion.loaders import load_document  # noqa: E402
from src.retrieval.index import HybridIndex  # noqa: E402

CORPUS_DIR = Path("eval/corpus")


def main():
    index = HybridIndex()
    for path in sorted(CORPUS_DIR.glob("*.md")):
        document = load_document(path)
        chunks = chunk_document(document, strategy="structure")
        result = index.add(chunks)
        print(f"{path.name}: {result.added} added, {len(result.skipped)} skipped")
    print(f"Total chunks indexed: {index.count()}")


if __name__ == "__main__":
    main()
