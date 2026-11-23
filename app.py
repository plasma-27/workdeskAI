from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from src.config import Settings, load_settings
from src.documents import chunk_uploaded_file, load_and_chunk_path
from src.rag import WorkDeskRAG
from src.schemas import RAGResult, ReferenceValidationError
from src.vector_store import VectorStore

SAMPLE_DATA_DIR = Path(__file__).resolve().parent / "data"

QUICK_QUESTIONS = [
    ("Password policy", "What are the password requirements?"),
    ("Email MFA (rephrased)", "Do I need a second step to log in to email?"),
    ("Missing info check", "What is the annual leave policy?"),
    (
        "Multi-doc query",
        "What are the password length requirements and which VPN client is required for remote work?",
    ),
]


st.set_page_config(
    page_title="WorkDesk AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_services() -> tuple[Settings, VectorStore, WorkDeskRAG]:
    settings = load_settings()
    store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )
    rag = WorkDeskRAG(settings=settings, vector_store=store)
    return settings, store, rag


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def render_assistant_details(payload: dict) -> None:
    """Render answer, reference badges, insufficient_context flag, chunks, and raw JSON."""
    response = payload["response"]
    retrieved_chunks = payload["retrieved_chunks"]

    st.markdown(response["answer"])

    if response["insufficient_context"]:
        st.warning(
            "Insufficient context (`insufficient_context: true`) — "
            "the indexed documents do not contain enough evidence to answer this question."
        )
    else:
        refs = response.get("source_references", [])
        if refs:
            badges = " ".join(f"`{ref}`" for ref in refs)
            st.caption(f"Supported by: {badges}")

    cited_set = set(response.get("source_references", []))

    with st.expander(f"Retrieved Chunks ({len(retrieved_chunks)})", expanded=False):
        if not retrieved_chunks:
            st.info("No chunks were retrieved.")
        else:
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                cited_tag = " · **CITED**" if chunk["reference"] in cited_set else ""
                st.markdown(
                    f"**#{rank} `{chunk['reference']}`** — "
                    f"cosine distance: `{chunk['distance']:.4f}`{cited_tag}"
                )
                st.text(chunk["text"])
                if rank < len(retrieved_chunks):
                    st.divider()

    with st.expander("Validated Pydantic JSON", expanded=False):
        st.json(response)


def render_sidebar(settings: Settings, store: VectorStore, rag: WorkDeskRAG) -> int:
    with st.sidebar:
        st.title("WorkDesk AI")
        st.caption("Document-grounded RAG · OpenAI SDK + ChromaDB + Pydantic")

        # 1. API / Model Status
        with st.expander("API Configuration (.env)", expanded=False):
            st.markdown(f"**Model:** `{settings.model}`")
            st.markdown(f"**Base URL:** `{settings.base_url}`")
            st.markdown(f"**Collection:** `{settings.chroma_collection}`")
            if st.button("Test API Connection", width="stretch"):
                with st.spinner(f"Pinging {settings.model}..."):
                    try:
                        reply = rag.ping()
                        st.success(reply)
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

        st.divider()

        # 2. Indexing Parameters
        st.subheader("Indexing & Retrieval")
        chunk_size = st.number_input(
            "Chunk size (words)",
            min_value=30,
            max_value=1000,
            value=settings.chunk_size_words,
            step=10,
        )
        max_overlap = max(0, int(chunk_size) - 1)
        default_overlap = min(settings.chunk_overlap_words, max_overlap)
        chunk_overlap = st.number_input(
            "Chunk overlap (words)",
            min_value=0,
            max_value=max_overlap,
            value=default_overlap,
            step=5,
        )
        top_k = st.slider(
            "Top-K retrieved chunks",
            min_value=1,
            max_value=10,
            value=min(settings.top_k, 10),
        )

        st.divider()

        # 3. Document Upload & Indexing
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, or TXT (selectable text only)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )

        if st.button(
            "Index Uploaded Files",
            type="primary",
            disabled=not uploaded_files,
            width="stretch",
        ):
            indexed_total = 0
            skipped_files: list[str] = []

            with st.spinner("Extracting, chunking, and embedding..."):
                for uf in uploaded_files:
                    try:
                        file_chunks = chunk_uploaded_file(
                            filename=uf.name,
                            data=uf.getvalue(),
                            chunk_size=int(chunk_size),
                            overlap=int(chunk_overlap),
                        )
                    except Exception as exc:
                        st.error(f"Failed to read '{uf.name}': {exc}")
                        continue

                    if not file_chunks:
                        skipped_files.append(uf.name)
                        continue

                    # Replace older chunks from the same filename cleanly before upserting
                    store.delete_source(Path(uf.name).name)
                    indexed_total += store.upsert_chunks(file_chunks)

            st.session_state.uploader_key += 1
            if indexed_total > 0:
                st.success(f"Indexed {indexed_total} chunk(s).")
            if skipped_files:
                st.warning(
                    "No selectable text extracted from: "
                    + ", ".join(skipped_files)
                    + " (scanned PDFs require OCR)."
                )
            st.rerun()

        col_sample, col_reset = st.columns(2)
        with col_sample:
            if st.button("Load Sample Docs", width="stretch"):
                if not SAMPLE_DATA_DIR.exists():
                    st.error(f"Sample directory not found: {SAMPLE_DATA_DIR}")
                else:
                    with st.spinner("Indexing data/..."):
                        chunks = load_and_chunk_path(
                            target=SAMPLE_DATA_DIR,
                            chunk_size=int(chunk_size),
                            overlap=int(chunk_overlap),
                        )
                        stored = store.upsert_chunks(chunks)
                    st.success(f"Indexed {stored} sample chunk(s).")
                    st.rerun()

        with col_reset:
            if st.button("Reset DB", width="stretch"):
                store.reset()
                st.session_state.messages = []
                st.rerun()

        st.divider()

        # 4. Indexed Sources Manager
        sources = store.list_sources()
        st.subheader(f"Indexed Documents ({len(sources)})")
        if not sources:
            st.caption("No documents indexed yet. Upload files or load sample docs.")
        else:
            for source_name, chunk_count in sources.items():
                c_name, c_del = st.columns([4, 1], vertical_alignment="center")
                with c_name:
                    st.markdown(f"`{source_name}` ({chunk_count} chunks)")
                with c_del:
                    if st.button("✕", key=f"del_{source_name}", help=f"Remove {source_name}"):
                        store.delete_source(source_name)
                        st.rerun()

        return top_k


def run_qa_turn(rag: WorkDeskRAG, question: str, top_k: int) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and validating response..."):
            try:
                result: RAGResult = rag.ask(question=question, top_k=top_k)
                payload = {
                    "response": result.response.model_dump(),
                    "retrieved_chunks": [c.model_dump() for c in result.retrieved_chunks],
                }
                render_assistant_details(payload)
                st.session_state.messages.append({"role": "assistant", "payload": payload})
            except (ReferenceValidationError, ValidationError) as exc:
                err_msg = f"Validation failed: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "error", "content": err_msg})
            except Exception as exc:
                err_msg = f"Request failed: {exc}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "error", "content": err_msg})


def render_chat_view(store: VectorStore, rag: WorkDeskRAG, top_k: int) -> None:
    # Quick scenario buttons + chat utility actions
    action_cols = st.columns([1, 1, 1, 1, 1, 1])
    for idx, (label, prompt_text) in enumerate(QUICK_QUESTIONS):
        with action_cols[idx]:
            if st.button(label, width="stretch", key=f"quick_{idx}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

    with action_cols[4]:
        if st.button("Clear Chat", width="stretch", disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()

    with action_cols[5]:
        export_json = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="Export JSON",
            data=export_json,
            file_name="workdesk_chat_export.json",
            mime="application/json",
            width="stretch",
            disabled=not st.session_state.messages,
        )

    st.divider()

    if store.count() == 0:
        st.info(
            "Your vector database is currently empty. Upload PDF/DOCX/TXT files in the sidebar "
            "or click **Load Sample Docs** to begin."
        )

    # Render history
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif role == "assistant":
            with st.chat_message("assistant"):
                render_assistant_details(msg["payload"])
        elif role == "error":
            with st.chat_message("assistant"):
                st.error(msg["content"])

    # Handle either a queued quick-question click or fresh chat_input submission
    chat_prompt = st.chat_input("Ask a question about your indexed documents...")
    prompt_to_run = chat_prompt or st.session_state.pending_prompt
    if prompt_to_run:
        st.session_state.pending_prompt = None
        run_qa_turn(rag=rag, question=prompt_to_run.strip(), top_k=top_k)


def render_search_debugger(store: VectorStore, top_k: int) -> None:
    st.subheader("Semantic Retrieval Debugger")
    st.caption(
        "Inspect raw ChromaDB cosine distances before LLM generation. "
        "Lower cosine distance means closer semantic vectors, not guaranteed relevance."
    )

    if store.count() == 0:
        st.info("Index documents in the sidebar first to test semantic retrieval.")
        return

    query = st.text_input(
        "Test search query",
        placeholder="e.g., How do I recover my account access?",
    )
    if st.button("Run Semantic Search", type="primary", disabled=not query.strip()):
        results = store.query(question=query.strip(), top_k=top_k)
        if not results:
            st.warning("No matching chunks returned.")
            return

        rows = [
            {
                "Rank": rank,
                "Reference": chunk.reference,
                "Source": chunk.source,
                "Chunk Index": chunk.chunk_index,
                "Cosine Distance": round(chunk.distance, 4),
            }
            for rank, chunk in enumerate(results, start=1)
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

        for rank, chunk in enumerate(results, start=1):
            with st.expander(
                f"#{rank} {chunk.reference} (distance={chunk.distance:.4f})",
                expanded=(rank == 1),
            ):
                st.text(chunk.text)


def main() -> None:
    init_session_state()

    try:
        settings, store, rag = get_services()
    except ValueError as exc:
        st.error(str(exc))
        st.code(
            "cp .env.example .env\n"
            "# Set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL in .env",
            language="bash",
        )
        st.stop()

    top_k = render_sidebar(settings, store, rag)

    # Top KPI bar
    sources = store.list_sources()
    m1, m2, m3 = st.columns(3)
    m1.metric("Indexed Documents", len(sources))
    m2.metric("Total Chunks in ChromaDB", store.count())
    m3.metric("Active LLM", settings.model)

    mode = st.segmented_control(
        "View",
        options=["Q&A Chat", "Semantic Search Debugger"],
        default="Q&A Chat",
        label_visibility="collapsed",
    )

    if mode == "Semantic Search Debugger":
        render_search_debugger(store=store, top_k=top_k)
    else:
        render_chat_view(store=store, rag=rag, top_k=top_k)


if __name__ == "__main__":
    main()
