from __future__ import annotations

from collections import Counter
from pathlib import Path

import chromadb

from src.schemas import Chunk, RetrievedChunk


class VectorStore:
    """Persistent ChromaDB wrapper using local default embeddings and cosine distance."""

    def __init__(self, persist_dir: Path, collection_name: str = "workdesk_docs") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        # ChromaDB's DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX) is used automatically.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection for a clean re-index."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> dict[str, int]:
        """Return a sorted mapping of indexed source filenames to their chunk counts."""
        if self.count() == 0:
            return {}

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        counts = Counter(str(meta.get("source", "unknown")) for meta in metadatas if meta)
        return dict(sorted(counts.items()))

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a single source document."""
        existing = self.collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Store or update chunks, metadata, and IDs in ChromaDB."""
        if not chunks:
            return 0

        self.collection.upsert(
            ids=[chunk.reference for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source, "chunk_index": chunk.chunk_index}
                for chunk in chunks
            ],
        )
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Run semantic search and return the top_k nearest chunks by cosine distance."""
        total = self.count()
        if total == 0:
            return []

        n_results = min(top_k, total)
        raw = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw["ids"][0] if raw.get("ids") else []
        docs = raw["documents"][0] if raw.get("documents") else []
        metas = raw["metadatas"][0] if raw.get("metadatas") else []
        distances = raw["distances"][0] if raw.get("distances") else []

        results: list[RetrievedChunk] = []
        for ref_id, text, meta, dist in zip(ids, docs, metas, distances):
            results.append(
                RetrievedChunk(
                    reference=ref_id,
                    text=text,
                    source=str(meta.get("source", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    distance=float(dist),
                )
            )
        return results
