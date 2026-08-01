import json
from pathlib import Path

from src.ingestion.pipeline import ingest_file
from src.ingestion.store import PROCESSED_DIR

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_file_writes_processed_json():
    """Smoke test: loading + saving actually wires up end to end."""
    out_path = PROCESSED_DIR / "sample.txt.json"
    try:
        doc = ingest_file(FIXTURES / "sample.txt")
        assert doc.format == "txt"
        assert out_path.exists()

        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["format"] == "txt"
        assert saved["sections"][0]["text"] == doc.sections[0].text
    finally:
        out_path.unlink(missing_ok=True)
