"""Persists loaded Documents as JSON in data/processed/, keeping them separate
from the originals in data/raw/ so later pipeline stages can re-index without
re-parsing the source files.
"""
import json
from dataclasses import asdict
from pathlib import Path

from .models import Document

PROCESSED_DIR = Path("data/processed")


def save_processed(document: Document, processed_dir: Path = PROCESSED_DIR) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    # Use the full original filename (extension included) so e.g. sample.txt
    # and sample.md don't collide into the same sample.json.
    out_path = processed_dir / (Path(document.source_path).name + ".json")
    out_path.write_text(json.dumps(asdict(document), indent=2), encoding="utf-8")
    return out_path
