from pathlib import Path

import pytest

from src.ingestion.loaders import load_document

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_txt():
    doc = load_document(FIXTURES / "sample.txt")
    assert doc.format == "txt"
    assert len(doc.sections) == 1
    assert doc.sections[0].heading is None
    assert "plain text document" in doc.sections[0].text


def test_load_markdown_splits_by_heading():
    doc = load_document(FIXTURES / "sample.md")
    assert doc.format == "md"
    headings = [s.heading for s in doc.sections]
    assert headings == ["Introduction", "Details"]
    assert "intro section" in doc.sections[0].text
    assert "details section" in doc.sections[1].text


def test_load_html_splits_by_heading():
    doc = load_document(FIXTURES / "sample.html")
    assert doc.format == "html"
    headings = [s.heading for s in doc.sections]
    assert headings == ["Welcome", "Section Two"]
    assert "welcome paragraph" in doc.sections[0].text


def test_load_pdf_splits_by_page():
    doc = load_document(FIXTURES / "sample.pdf")
    assert doc.format == "pdf"
    assert len(doc.sections) == 2
    assert doc.sections[0].page_number == 1
    assert doc.sections[1].page_number == 2
    assert "page one" in doc.sections[0].text
    assert "page two" in doc.sections[1].text


def test_unsupported_format_raises(tmp_path):
    bad_file = tmp_path / "sample.docx"
    bad_file.write_text("not supported")
    with pytest.raises(ValueError):
        load_document(bad_file)
