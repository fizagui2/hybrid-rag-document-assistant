"""Pydantic request/response models for the API.

Response models mirror the dataclasses already used throughout the
pipeline (Answer, RerankedHit, VerifiedCitation, ConfidenceScore) rather
than redefining the data — model_validate(..., from_attributes) builds a
Pydantic model directly from a dataclass instance, so there's one source
of truth for what an "answer" actually contains.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    final_k: int = Field(default=5, ge=1, le=20)


class SourceChunk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    text: str
    metadata: dict
    rerank_score: float
    fused_score: float


class Citation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    claim: str
    citation_number: int
    supported: bool
    reasoning: str


class Confidence(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


class AskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str
    text: str
    sources: list[SourceChunk]
    citations: list[Citation]
    confidence: Optional[Confidence]
    low_confidence: bool


class DocumentsResponse(BaseModel):
    documents: list[str]
    chunk_count: int


class IngestedFile(BaseModel):
    filename: str
    chunks_added: int
    chunks_skipped: int


class IngestResponse(BaseModel):
    strategy: str
    files: list[IngestedFile]
