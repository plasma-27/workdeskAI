from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import load_settings
from src.documents import load_and_chunk_path
from src.rag import WorkDeskRAG
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _print_result(result, show_chunks: bool = False, as_json: bool = False) -> None:
    if as_json:
        print(result.response.model_dump_json(indent=2))
        return

    print("\nANSWER:")
    print(result.response.answer)
    print(f"\nSOURCE REFERENCES: {result.response.source_references}")
    print(f"INSUFFICIENT CONTEXT: {result.response.insufficient_context}")

    if show_chunks:
        print("\nRETRIEVED CHUNKS:")
        for rank, chunk in enumerate(result.retrieved_chunks, start=1):
            preview = chunk.text[:180].replace("\n", " ")
            print(
                f"  [{rank}] {chunk.reference} "
                f"(cosine distance={chunk.distance:.4f}) -> {preview}..."
            )


def cmd_ping() -> int:
    settings = load_settings()
    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    rag = WorkDeskRAG(settings, store)
    print(f"Base URL : {settings.base_url}")
    print(f"Model    : {settings.model}")
    print(f"Response : {rag.ping()}")
    return 0


def cmd_index(path_str: str, reset: bool) -> int:
    settings = load_settings()
    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)

    if reset:
        store.reset()
        print(f"Cleared collection '{settings.chroma_collection}'.")

    target = Path(path_str)
    chunks = load_and_chunk_path(
        target=target,
        chunk_size=settings.chunk_size_words,
        overlap=settings.chunk_overlap_words,
    )
    stored = store.upsert_chunks(chunks)
    print(
        f"Indexed {stored} chunk(s) from '{target}' into "
        f"'{settings.chroma_collection}' (total chunks in DB: {store.count()})."
    )
    return 0


def cmd_ask(question: str, top_k: int | None, show_chunks: bool, as_json: bool) -> int:
    settings = load_settings()
    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    rag = WorkDeskRAG(settings, store)
    result = rag.ask(question=question, top_k=top_k)
    _print_result(result, show_chunks=show_chunks, as_json=as_json)
    return 0


def cmd_chat(top_k: int | None, show_chunks: bool) -> int:
    settings = load_settings()
    store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    rag = WorkDeskRAG(settings, store)

    print(f"WorkDesk AI interactive session (model={settings.model}, indexed chunks={store.count()})")
    print("Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return 0

        try:
            result = rag.ask(question=question, top_k=top_k)
            _print_result(result, show_chunks=show_chunks)
            print()
        except Exception as exc:
            print(f"ERROR: {exc}\n", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workdesk",
        description="WorkDesk AI: Universal Document Intelligence Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping", help="Verify OpenAI SDK connectivity to Gemini via .env")

    p_index = sub.add_parser("index", help="Load, chunk, embed, and store documents in ChromaDB")
    p_index.add_argument("path", nargs="?", default="data", help="File or directory to index (default: data)")
    p_index.add_argument("--reset", action="store_true", help="Clear existing collection before indexing")

    p_ask = sub.add_parser("ask", help="Ask a single question against indexed documents")
    p_ask.add_argument("question", help="Natural-language question")
    p_ask.add_argument("--top-k", type=int, default=None, help="Override TOP_K retrieved chunks")
    p_ask.add_argument("--show-chunks", action="store_true", help="Display retrieved chunks and cosine distances")
    p_ask.add_argument("--json", dest="as_json", action="store_true", help="Print raw validated JSON")

    p_chat = sub.add_parser("chat", help="Start an interactive Q&A session")
    p_chat.add_argument("--top-k", type=int, default=None, help="Override TOP_K retrieved chunks")
    p_chat.add_argument("--show-chunks", action="store_true", help="Display retrieved chunks and cosine distances")

    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "ping":
        return cmd_ping()
    if args.command == "index":
        return cmd_index(args.path, args.reset)
    if args.command == "ask":
        return cmd_ask(args.question, args.top_k, args.show_chunks, args.as_json)
    if args.command == "chat":
        return cmd_chat(args.top_k, args.show_chunks)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
