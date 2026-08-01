"""Chunking strategies: turn a Document's sections into embedding-sized Chunks.

Three strategies live here:
- "fixed"     — naive sliding window over the raw text, ignores structure.
- "structure" — respects the Document's existing Section boundaries, splitting
                 only sections that are too long, via recursive character
                 splitting (paragraph -> line -> sentence -> word -> hard cut).
- "semantic"  — splits sentences into groups based on embedding-similarity
                 topic shifts, rather than structure or raw size.

chunk_size / overlap are measured in characters, a simple stand-in for token
count that needs no extra dependency. Swapping in a real tokenizer later
only means changing how length is measured, not the algorithms below.
"""
from typing import Optional

import numpy as np

from ..embeddings import embed_texts
from ..text import split_sentences
from .models import Chunk, Document, Section

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_BREAKPOINT_PERCENTILE = 95

_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _section_offsets(document: Document, separator: str = "\n\n") -> list[tuple[int, int, Section]]:
    """Character offset ranges for each section within document.full_text,
    matching how Document.full_text joins sections together.
    """
    offsets = []
    cursor = 0
    for section in document.sections:
        start = cursor
        end = start + len(section.text)
        offsets.append((start, end, section))
        cursor = end + len(separator)
    return offsets


def _section_at(offsets: list[tuple[int, int, Section]], position: int) -> Optional[Section]:
    for start, end, section in offsets:
        if start <= position < end:
            return section
    return offsets[-1][2] if offsets else None


def chunk_fixed(document: Document, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Baseline strategy: slide a fixed-size window with overlap over the
    document's full text, ignoring section boundaries. A chunk can span two
    sections; metadata is attributed using whichever section it starts in.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.full_text
    offsets = _section_offsets(document)
    step = chunk_size - overlap

    chunks = []
    position = 0
    index = 0
    while position < len(text):
        window = text[position: position + chunk_size].strip()
        if window:
            section = _section_at(offsets, position)
            chunks.append(Chunk(
                text=window,
                source_path=document.source_path,
                chunk_index=index,
                strategy="fixed",
                heading=section.heading if section else None,
                page_number=section.page_number if section else None,
            ))
            index += 1
        position += step
    return chunks


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Break text into pieces that each fit within chunk_size, trying
    separators from most to least meaningful. Each piece keeps its trailing
    separator (except the last), so punctuation/whitespace consumed by
    str.split() isn't lost when pieces are reassembled later. Falls back to
    a hard character cut once no separators are left.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, *remaining = separators
    raw_pieces = text.split(separator)
    # str.split() consumes the separator; re-attach it to every piece but
    # the last so e.g. "one. two" -> ["one. ", "two"], not ["one", "two"].
    pieces = [p + separator for p in raw_pieces[:-1]] + raw_pieces[-1:]

    result = []
    for piece in pieces:
        if not piece.strip():
            continue
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(_split_recursive(piece, chunk_size, remaining))
    return result


def _merge_small_pieces(pieces: list[str], chunk_size: int) -> list[str]:
    """Greedily merge adjacent small pieces up to chunk_size, so we don't end
    up with a pile of tiny fragments. Concatenation is direct (no inserted
    separator) because each piece from _split_recursive already carries its
    own trailing whitespace/punctuation — pieces joined in order reconstruct
    the section's real content. Deliberately left *unstripped*: _apply_overlap
    needs each piece's real trailing characters (including boundary spaces).
    """
    merged = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                merged.append(current)
            current = piece
    if current.strip():
        merged.append(current)
    return merged


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Stitch trailing-text overlap between adjacent (unstripped) pieces,
    then strip each result's outer edges. Because adjacent pieces are
    contiguous slices of the original text, one piece's tail concatenated
    onto the next is itself a valid, correctly-spaced (if redundant) slice —
    no extra spacing needs to be inserted.
    """
    if overlap > 0 and len(pieces) >= 2:
        overlapped = [pieces[0]]
        for i in range(1, len(pieces)):
            carry = pieces[i - 1][-overlap:]
            overlapped.append(f"{carry}{pieces[i]}")
        pieces = overlapped
    return [p.strip() for p in pieces]


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Merge small pieces up to chunk_size, then apply overlap between them."""
    return _apply_overlap(_merge_small_pieces(pieces, chunk_size), overlap)


def chunk_structure(document: Document, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Structure-aware strategy: keep short sections whole; recursively split
    (and then re-merge with overlap) sections that exceed chunk_size.
    """
    chunks = []
    index = 0
    for section in document.sections:
        pieces = _split_recursive(section.text, chunk_size, _RECURSIVE_SEPARATORS)
        pieces = _merge_pieces(pieces, chunk_size, overlap)
        for piece in pieces:
            chunks.append(Chunk(
                text=piece,
                source_path=document.source_path,
                chunk_index=index,
                strategy="structure",
                heading=section.heading,
                page_number=section.page_number,
            ))
            index += 1
    return chunks


def _group_by_breakpoints(sentences: list[str], distances: list[float], threshold: float) -> list[str]:
    """Group consecutive sentences into segments, cutting a new segment
    wherever the distance to the next sentence exceeds threshold (a topic
    shift), and re-joining each segment's sentences with a single space.
    """
    segments = []
    current = [sentences[0]]
    for i, distance in enumerate(distances):
        if distance > threshold:
            segments.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    segments.append(" ".join(current))
    return segments


def chunk_semantic(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    breakpoint_percentile: float = DEFAULT_BREAKPOINT_PERCENTILE,
) -> list[Chunk]:
    """Semantic strategy: split each section into sentences, embed them, and
    cut a new chunk wherever adjacent sentences are unusually dissimilar —
    an adaptive threshold (a percentile of that section's own distance
    distribution) rather than one fixed magic number. Segments that still
    exceed chunk_size are further split with the same recursive splitter
    structure-aware chunking uses; segments are never merged across a
    detected topic boundary just to hit a size target.
    """
    chunks = []
    index = 0
    for section in document.sections:
        sentences = split_sentences(section.text)

        if len(sentences) <= 1:
            segments = [section.text.strip()] if section.text.strip() else []
        else:
            vectors = embed_texts(sentences)
            distances = [float(1 - np.dot(vectors[i], vectors[i + 1])) for i in range(len(vectors) - 1)]
            threshold = float(np.percentile(distances, breakpoint_percentile))
            segments = _group_by_breakpoints(sentences, distances, threshold)

        # Cap segment size regardless of which branch produced them above —
        # a lone oversized sentence needs splitting just as much as an
        # oversized multi-sentence topic segment does.
        capped_segments = []
        for segment in segments:
            if len(segment) <= chunk_size:
                capped_segments.append(segment)
            else:
                pieces = _split_recursive(segment, chunk_size, _RECURSIVE_SEPARATORS)
                capped_segments.extend(_merge_small_pieces(pieces, chunk_size))
        segments = capped_segments

        for piece in _apply_overlap(segments, overlap):
            chunks.append(Chunk(
                text=piece,
                source_path=document.source_path,
                chunk_index=index,
                strategy="semantic",
                heading=section.heading,
                page_number=section.page_number,
            ))
            index += 1
    return chunks


_STRATEGIES = {
    "fixed": chunk_fixed,
    "structure": chunk_structure,
    "semantic": chunk_semantic,
}


def chunk_document(
    document: Document,
    strategy: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Dispatch to the requested chunking strategy."""
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy!r} (available: {sorted(_STRATEGIES)})")
    return _STRATEGIES[strategy](document, chunk_size=chunk_size, overlap=overlap)
