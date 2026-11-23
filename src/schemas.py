from __future__ import annotations

import re

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A single word-bounded chunk extracted from a source document."""

    reference: str = Field(..., description="Unique ID in '<source>#chunk-<index>' format.")
    text: str = Field(..., min_length=1)
    source: str
    chunk_index: int = Field(..., ge=0)


class RetrievedChunk(Chunk):
    """A chunk returned by ChromaDB semantic search along with its cosine distance."""

    distance: float


class WorkDeskResponse(BaseModel):
    """Structured RAG response schema validated via Pydantic."""

    answer: str = Field(..., min_length=1)
    source_references: list[str] = Field(default_factory=list)
    insufficient_context: bool


class RAGResult(BaseModel):
    """Full pipeline result bundling the validated answer and retrieved evidence."""

    question: str
    response: WorkDeskResponse
    retrieved_chunks: list[RetrievedChunk]


class ReferenceValidationError(ValueError):
    """Raised when the LLM cites references not present in the retrieved context."""


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_PATTERN.sub("", cleaned).strip()
    return cleaned


def validate_rag_response(
    raw_json: str,
    retrieved_chunks: list[RetrievedChunk],
) -> WorkDeskResponse:
    """Validate raw JSON against WorkDeskResponse and verify all cited references."""
    if not raw_json or not raw_json.strip():
        raise ValueError("LLM returned an empty response.")

    cleaned = _strip_code_fences(raw_json)
    parsed = WorkDeskResponse.model_validate_json(cleaned)

    # Deduplicate references while preserving order
    deduped_refs = list(dict.fromkeys(ref.strip() for ref in parsed.source_references if ref.strip()))
    parsed = parsed.model_copy(update={"source_references": deduped_refs})

    valid_refs = {chunk.reference for chunk in retrieved_chunks}
    invalid_refs = [ref for ref in parsed.source_references if ref not in valid_refs]

    if invalid_refs:
        raise ReferenceValidationError(
            f"Model cited reference(s) not present in retrieved chunks: {invalid_refs}. "
            f"Valid retrieved references: {sorted(valid_refs)}"
        )

    if not parsed.insufficient_context and not parsed.source_references:
        raise ReferenceValidationError(
            "Model set insufficient_context=false but returned an empty source_references list."
        )

    return parsed
