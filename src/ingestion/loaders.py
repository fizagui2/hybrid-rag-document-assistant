"""Format-specific loaders that turn raw files into normalized Document objects.

Each loader takes a file path and returns a Document with one Section per
logical piece of content (a heading's body for md/html, a page for pdf, the
whole file for txt).
"""
import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from .models import Document, Section

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def load_txt(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    return Document(source_path=str(path), format="txt", sections=[Section(text=text)])


def load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    headings = list(_MD_HEADING_RE.finditer(raw))

    if not headings:
        return Document(source_path=str(path), format="md", sections=[Section(text=raw.strip())])

    sections = []

    # Content before the first heading (if any) is kept as an untitled section.
    preamble = raw[: headings[0].start()].strip()
    if preamble:
        sections.append(Section(text=preamble))

    for i, match in enumerate(headings):
        heading_text = match.group(2).strip()
        body_start = match.end()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        body = raw[body_start:body_end].strip()
        sections.append(Section(text=body, heading=heading_text))

    return Document(source_path=str(path), format="md", sections=sections)


def load_html(path: Path) -> Document:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    sections: list[Section] = []
    current_heading: str | None = None
    current_parts: list[str] = []

    def flush():
        text = "\n".join(current_parts).strip()
        if text:
            sections.append(Section(text=text, heading=current_heading))

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if element.name.startswith("h"):
            flush()
            current_heading = element.get_text(strip=True)
            current_parts = []
        else:
            piece = element.get_text(strip=True)
            if piece:
                current_parts.append(piece)
    flush()

    if not sections:
        # No h*/p/li tags found at all — fall back to grabbing everything.
        text = soup.get_text(separator="\n", strip=True)
        if text:
            sections = [Section(text=text)]

    return Document(source_path=str(path), format="html", sections=sections)


def load_pdf(path: Path) -> Document:
    reader = PdfReader(str(path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text=text, page_number=page_number))
    return Document(source_path=str(path), format="pdf", sections=sections)


_LOADERS = {
    ".txt": load_txt,
    ".md": load_markdown,
    ".html": load_html,
    ".htm": load_html,
    ".pdf": load_pdf,
}

SUPPORTED_EXTENSIONS = frozenset(_LOADERS)


def load_document(path: str | Path) -> Document:
    """Dispatch to the right loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"Unsupported file type: {suffix!r} (supported: {sorted(SUPPORTED_EXTENSIONS)})")
    return _LOADERS[suffix](path)
