from __future__ import annotations

import logging
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from src.schemas import Chunk

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def extract_text(file_path: Path) -> str:
    """Extract plain text from TXT, PDF (selectable text), or DOCX files."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="replace").strip()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)
        extracted = "\n\n".join(pages).strip()
        if not extracted:
            logger.warning(
                "No selectable text extracted from '%s'. "
                "Scanned PDFs require OCR, which is not implemented.",
                file_path.name,
            )
        return extracted

    if suffix == ".docx":
        doc = DocxDocument(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paragraphs).strip()

    raise ValueError(
        f"Unsupported file type '{suffix}' for '{file_path}'. "
        f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if not (0 <= overlap < chunk_size):
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                reference=f"{source}#chunk-{chunk_index}",
                text=" ".join(chunk_words),
                source=source,
                chunk_index=chunk_index,
            )
        )
        if end == len(words):
            break
        start += step
        chunk_index += 1

    return chunks


def discover_files(target: Path) -> list[Path]:
    """Return a sorted list of supported document paths from a file or directory."""
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {target}")

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {target.suffix}")
        return [target]

    files = [
        p
        for p in sorted(target.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def load_and_chunk_path(
    target: Path,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[Chunk]:
    """Load all supported documents at `target` and split them into overlapping chunks."""
    files = discover_files(target)
    base_dir = target if target.is_dir() else target.parent
    all_chunks: list[Chunk] = []

    for file_path in files:
        source_label = file_path.relative_to(base_dir).as_posix()
        text = extract_text(file_path)
        if not text:
            logger.warning("Skipping '%s' because no text could be extracted.", source_label)
            continue
        file_chunks = chunk_text(
            text=text,
            source=source_label,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(file_chunks)

    return all_chunks
