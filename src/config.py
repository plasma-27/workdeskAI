from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    chroma_persist_dir: Path
    chroma_collection: str
    chunk_size_words: int
    chunk_overlap_words: int
    top_k: int


def load_settings(env_path: str | Path | None = None) -> Settings:
    """Load and validate application configuration from .env and environment variables."""
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()

    missing = []
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not model:
        missing.append("OPENAI_MODEL")

    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and populate them."
        )

    chunk_size = int(os.getenv("CHUNK_SIZE_WORDS", "180"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))
    top_k = int(os.getenv("TOP_K", "3"))

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE_WORDS must be greater than 0.")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError("CHUNK_OVERLAP_WORDS must be >= 0 and < CHUNK_SIZE_WORDS.")
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than 0.")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        chroma_persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "workdesk_docs").strip(),
        chunk_size_words=chunk_size,
        chunk_overlap_words=chunk_overlap,
        top_k=top_k,
    )
