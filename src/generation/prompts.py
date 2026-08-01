"""Builds the grounded-generation prompt: system instructions plus a
numbered context block assembled from retrieved chunks, so the model can be
told to cite by number and the numbers actually correspond to something.
"""
from ..retrieval.reranker import RerankedHit

SYSTEM_PROMPT = """You are a document assistant that answers questions using ONLY the provided context.

Rules:
- Only use information found in the numbered context blocks below.
- Cite the specific context block(s) that support each claim using bracketed numbers, e.g. [1] or [1][2].
- If the context does not contain enough information to answer the question, say so explicitly instead of guessing.
- Do not use any outside knowledge, even if you believe it to be true."""


def format_context(chunks: list[RerankedHit]) -> str:
    """Turn retrieved chunks into a numbered context block. The numbers
    here are exactly the numbers the model is instructed to cite with.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source_path", "unknown")
        heading = chunk.metadata.get("heading")
        page = chunk.metadata.get("page_number")

        location = source
        if heading:
            location += f" — {heading}"
        if page is not None and page != -1:
            location += f" (page {page})"

        blocks.append(f"[{i}] (source: {location})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RerankedHit]) -> list[dict]:
    """Build the full chat message list to send to the LLM."""
    context = format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
