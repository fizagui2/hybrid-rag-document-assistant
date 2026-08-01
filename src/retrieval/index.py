"""Hybrid index: a persistent Chroma vector store (dense) plus a BM25
keyword index (sparse), both built from the same set of Chunks.

Both indexes are kept in sync by construction:
- Chunk IDs are deterministic (source_path + strategy + chunk_index), so
  re-indexing the same chunk *updates* it in both indexes instead of
  duplicating it (Chroma via upsert; BM25 via a dict keyed by chunk id).
- BM25 has no incremental-update API, so the tokenized corpus is kept as a
  dict (source of truth) and the BM25Okapi object is rebuilt from it after
  every add() — cheap at the corpus sizes this project deals with.

add() also rejects near-duplicate chunks (cosine similarity above a
threshold against anything already indexed, including earlier chunks from
the same call) — see _find_near_duplicate.

search_dense() / search_sparse() run the two retrieval methods individually;
fusing their results (Reciprocal Rank Fusion) and reranking live in
search.py, since those are pure algorithms over ranked lists and don't need
direct access to either backend.
"""
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

import chromadb

from ..embeddings import embed_query, embed_texts
from ..ingestion.models import Chunk

CHROMA_DIR = Path("data/chroma")
BM25_PATH = Path("data/bm25_index.pkl")
COLLECTION_NAME = "chunks"
DEFAULT_DEDUPE_THRESHOLD = 0.95


@dataclass
class SkippedDuplicate:
    chunk_id: str
    duplicate_of: str
    similarity: float


@dataclass
class IndexResult:
    added: int
    skipped: list[SkippedDuplicate] = field(default_factory=list)


@dataclass
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25. Uses a \\w+ regex rather than
    text.split() so trailing punctuation isn't glued onto the token — a
    plain .split() turns "ERR_502." (period attached, from a sentence) into
    a token that will never match a query's "ERR_502", silently breaking
    exact-term matches at sentence boundaries.
    """
    return _WORD_RE.findall(text.lower())


def _chunk_metadata(chunk: Chunk) -> dict:
    """Chroma metadata values must be str/int/float/bool — None isn't
    allowed — so optional fields get sentinel values instead.
    """
    return {
        "source_path": chunk.source_path,
        "chunk_index": chunk.chunk_index,
        "strategy": chunk.strategy,
        "heading": chunk.heading or "",
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "char_count": chunk.char_count,
    }


class HybridIndex:
    def __init__(self, chroma_dir: Path = CHROMA_DIR, bm25_path: Path = BM25_PATH):
        self._bm25_path = Path(bm25_path)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        # "cosine" space makes Chroma report distance = 1 - cosine_similarity
        # directly (verified against known vectors), which is what the
        # dedup check below needs. Only takes effect the first time this
        # collection is created — an existing collection keeps whatever
        # space it was created with.
        self._collection = self._client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

        self._bm25_tokenized_by_id: dict[str, list[str]] = {}
        self._bm25: Optional[BM25Okapi] = None
        self._load_bm25()
        self._rebuild_bm25()

    @staticmethod
    def chunk_id(chunk: Chunk) -> str:
        return f"{chunk.source_path}::{chunk.strategy}::{chunk.chunk_index}"

    def add(self, chunks: list[Chunk], dedupe_threshold: float = DEFAULT_DEDUPE_THRESHOLD) -> IndexResult:
        """Embed and store chunks in both indexes, skipping any chunk that's
        a near-duplicate (cosine similarity > dedupe_threshold) of something
        already indexed — including chunks inserted earlier in this same
        call. Chunks are inserted one at a time (rather than one batch
        upsert) specifically so each new chunk's duplicate check sees
        everything indexed so far, including its own batch.
        """
        if not chunks:
            return IndexResult(added=0)

        ids = [self.chunk_id(c) for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        skipped = []
        inserted_ids = []
        for chunk, chunk_id, text, embedding in zip(chunks, ids, texts, embeddings):
            duplicate = self._find_near_duplicate(embedding, dedupe_threshold, exclude_id=chunk_id)
            if duplicate is not None:
                duplicate_of, similarity = duplicate
                skipped.append(SkippedDuplicate(chunk_id=chunk_id, duplicate_of=duplicate_of, similarity=similarity))
                continue

            self._collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                documents=[text],
                metadatas=[_chunk_metadata(chunk)],
            )
            self._bm25_tokenized_by_id[chunk_id] = _tokenize(text)
            inserted_ids.append(chunk_id)

        if inserted_ids:
            self._rebuild_bm25()
            self._save_bm25()

        return IndexResult(added=len(inserted_ids), skipped=skipped)

    def _find_near_duplicate(
        self, embedding, threshold: float, exclude_id: str
    ) -> Optional[tuple[str, float]]:
        """Return (existing_chunk_id, similarity) for the closest *other*
        already-indexed chunk if it's above threshold, else None.
        exclude_id matters because a chunk being re-indexed (same
        deterministic id, e.g. updated text) will otherwise find its own
        previous version as a near-perfect "duplicate" and get skipped
        instead of properly updated via upsert.
        """
        if self._collection.count() == 0:
            return None

        result = self._collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=2,  # ask for 2 in case the closest match is the chunk's own existing entry
            include=["distances"],
        )
        candidate_ids = result["ids"][0]
        distances = result["distances"][0]

        for candidate_id, distance in zip(candidate_ids, distances):
            if candidate_id == exclude_id:
                continue
            similarity = 1 - distance
            return (candidate_id, similarity) if similarity > threshold else None
        return None

    def search_dense(self, query: str, k: int = 10) -> list[SearchHit]:
        """Embed the query and return the k nearest chunks by cosine
        similarity. Uses embed_query() (not embed_texts()) since bge models
        expect an instruction prefix on the query side only.
        """
        if self.count() == 0:
            return []

        query_embedding = embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )

        return [
            SearchHit(chunk_id=chunk_id, text=text, metadata=metadata, score=1 - distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def search_sparse(self, query: str, k: int = 10) -> list[SearchHit]:
        """Score every chunk against the query with BM25 and return the top
        k with a positive score. Chunks sharing zero query terms score <= 0
        (see the negative-IDF note from the dedup milestone) and are
        dropped rather than padding out a "top k" with irrelevant results.
        """
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.bm25_ids, scores), key=lambda pair: pair[1], reverse=True)
        ranked = [(chunk_id, score) for chunk_id, score in ranked if score > 0][:k]
        if not ranked:
            return []

        records = self._collection.get(ids=[chunk_id for chunk_id, _ in ranked], include=["documents", "metadatas"])
        by_id = dict(zip(records["ids"], zip(records["documents"], records["metadatas"])))

        hits = []
        for chunk_id, score in ranked:
            if chunk_id not in by_id:
                continue
            text, metadata = by_id[chunk_id]
            hits.append(SearchHit(chunk_id=chunk_id, text=text, metadata=metadata, score=float(score)))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[str]:
        """Distinct source document filenames currently in the index."""
        if self.count() == 0:
            return []
        records = self._collection.get(include=["metadatas"])
        return sorted({Path(metadata["source_path"]).name for metadata in records["metadatas"]})

    @property
    def bm25_ids(self) -> list[str]:
        return list(self._bm25_tokenized_by_id.keys())

    def _rebuild_bm25(self) -> None:
        tokenized = list(self._bm25_tokenized_by_id.values())
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _save_bm25(self) -> None:
        self._bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._bm25_path, "wb") as f:
            pickle.dump(self._bm25_tokenized_by_id, f)

    def _load_bm25(self) -> None:
        if self._bm25_path.exists():
            with open(self._bm25_path, "rb") as f:
                self._bm25_tokenized_by_id = pickle.load(f)
