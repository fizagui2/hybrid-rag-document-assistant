"""Shared local embedding model wrapper.

Used by semantic chunking now (src/ingestion/chunker.py) and by dense
retrieval later (src/retrieval/). Runs entirely locally via
sentence-transformers — no API key, no per-request cost, no internet
dependency after the model has been downloaded once.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# bge-*-en-v1.5 models are trained asymmetrically: only the QUERY side of a
# search should carry this instruction prefix, never the indexed passages/
# chunks. Omitting it (or adding it to documents too) measurably hurts
# retrieval quality for this model family.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load and cache the embedding model so it's only loaded from disk once
    per process, no matter how many times embed_texts() is called.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]):
    """Embed passages/chunks (the side being *searched*) into normalized
    vectors, so cosine similarity between two embeddings reduces to a plain
    dot product. No instruction prefix — see embed_query() for why.
    """
    return get_embedder().encode(texts, normalize_embeddings=True)


def embed_query(query: str):
    """Embed a search query (the side doing the *searching*). Adds the
    instruction prefix bge-small-en-v1.5 expects on queries only.
    """
    return get_embedder().encode(_QUERY_INSTRUCTION + query, normalize_embeddings=True)
