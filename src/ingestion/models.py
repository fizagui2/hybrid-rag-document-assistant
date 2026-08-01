"""Data models shared by the ingestion pipeline."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Section:
    """A logical piece of a document: one heading's content, or one PDF page."""

    text: str
    heading: Optional[str] = None
    page_number: Optional[int] = None


@dataclass
class Document:
    """A fully loaded, normalized source document."""

    source_path: str
    format: str  # "txt" | "md" | "html" | "pdf"
    sections: list[Section] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(section.text for section in self.sections)


@dataclass
class Chunk:
    """A piece of a Document sized for embedding/retrieval, produced by one
    of the chunking strategies in chunker.py.
    """

    text: str
    source_path: str
    chunk_index: int
    strategy: str  # "fixed" | "structure" | "semantic"
    heading: Optional[str] = None
    page_number: Optional[int] = None
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)
