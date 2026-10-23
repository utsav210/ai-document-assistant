from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 60.0


def _init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []


def _check_backend_health() -> bool:
    """Check whether the FastAPI backend service is reachable."""
    try:
        response = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _index_document(file, session_id: str) -> dict[str, Any] | None:
    """Upload a document to the backend /documents/index endpoint."""
    try:
        files = {
            "file": (
                file.name,
                file.getvalue(),
                file.type or "application/octet-stream",
            )
        }
        data = {"session_id": session_id}
        response = httpx.post(
            f"{BACKEND_URL}/documents/index",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()

        detail = response.json().get("detail", response.text)
        st.sidebar.error(f"Indexing failed ({response.status_code}): {detail}")
        return None
    except httpx.ConnectError:
        st.sidebar.error("Cannot connect to backend server. Make sure it is running.")
        return None
    except (httpx.HTTPError, OSError) as exc:
        st.sidebar.error(f"Upload error: {exc}")
        return None


def _send_chat_message(
    session_id: str,
    question: str,
    source_name: str | None = None,
) -> dict[str, Any] | None:
    """Send a chat question to the backend /chat endpoint."""
    payload: dict[str, Any] = {
        "session_id": session_id,
        "question": question,
    }
    if source_name and source_name != "All Documents":
        payload["source_name"] = source_name

    try:
        response = httpx.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json()

        detail = response.json().get("detail", response.text)
        st.error(f"Chat request failed ({response.status_code}): {detail}")
        return None
    except httpx.ConnectError:
        st.error("Cannot connect to backend server. Make sure it is running.")
        return None
    except (httpx.HTTPError, OSError) as exc:
        st.error(f"Chat request error: {exc}")
        return None


def _clear_chat_history(session_id: str) -> bool:
    """Call backend to clear conversational memory for the session."""
    try:
        response = httpx.delete(
            f"{BACKEND_URL}/chat/history/{session_id}",
            timeout=5.0,
        )
        return response.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _fetch_backend_history(session_id: str) -> list[dict[str, str]]:
    """Fetch active 3-turn memory window from backend."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/chat/history/{session_id}",
            timeout=3.0,
        )
        if response.status_code == 200:
            return response.json().get("turns", [])
    except (httpx.HTTPError, OSError):
        return []
    return []


def main() -> None:
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📄",
        layout="wide",
    )

    _init_session_state()

    st.title("📄 AI Document Assistant")
    st.caption(
        "Upload PDF, DOCX, or TXT documents and ask questions. "
        "Supports conversational follow-ups remembering the latest 3 exchanges."
    )

    backend_ok = _check_backend_health()

    # --- Sidebar: Configuration & Document Management ---
    with st.sidebar:
        st.header("⚙️ Session & Documents")

        if backend_ok:
            st.success("Backend Connected", icon="🟢")
        else:
            st.error(
                f"Backend Disconnected ({BACKEND_URL}). "
                "Ensure `uvicorn backend.main:app` is running.",
                icon="🔴",
            )

        # Session ID controls
        st.subheader("Session")
        current_session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="Unique conversational session identifier. Retains up to 3 conversation turns.",
        )
        if (
            current_session_id != st.session_state.session_id
            and current_session_id.strip()
        ):
            st.session_state.session_id = current_session_id.strip()
            st.session_state.messages = []
            st.rerun()

        col_new, col_reset = st.columns(2)
        with col_new:
            if st.button("New Session", use_container_width=True):
                st.session_state.session_id = uuid.uuid4().hex[:12]
                st.session_state.messages = []
                st.session_state.indexed_documents = []
                st.rerun()

        with col_reset:
            if st.button("Reset Chat", use_container_width=True):
                _clear_chat_history(st.session_state.session_id)
                st.session_state.messages = []
                st.success("Chat reset.")
                st.rerun()

        st.divider()

        # Document Upload
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Select document",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF (with OCR/tables), DOCX, TXT. Max 20 MB.",
        )

        if uploaded_file is not None and st.button(
            "Index Document", type="primary", use_container_width=True
        ):
            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                result = _index_document(uploaded_file, st.session_state.session_id)
                if result:
                    doc_info = {
                        "name": result.get("source_name", uploaded_file.name),
                        "chunks": result.get("chunk_count", 0),
                        "method": result.get("extraction_method", "N/A"),
                        "pages": result.get("page_count"),
                    }
                    # Avoid duplicate listings in session UI
                    if not any(
                        d["name"] == doc_info["name"]
                        for d in st.session_state.indexed_documents
                    ):
                        st.session_state.indexed_documents.append(doc_info)
                    st.sidebar.success(
                        f"Indexed '{doc_info['name']}' ({doc_info['chunks']} chunks)"
                    )

        # Indexed documents in session
        if st.session_state.indexed_documents:
            st.subheader("Indexed Documents")
            for doc in st.session_state.indexed_documents:
                pages_str = f" | {doc['pages']} pages" if doc.get("pages") else ""
                st.caption(f"📄 **{doc['name']}** ({doc['chunks']} chunks{pages_str})")

        # Optional source filter
        source_options = ["All Documents"] + [
            doc["name"] for doc in st.session_state.indexed_documents
        ]
        selected_source = st.selectbox(
            "Query Scope",
            options=source_options,
            help="Filter questions to a specific uploaded document or search across all.",
        )

        st.divider()

        # Conversational memory inspection
        with st.expander("🧠 Active Memory Window (Last 3 Turns)", expanded=False):
            turns = _fetch_backend_history(st.session_state.session_id)
            if turns:
                for idx, turn in enumerate(turns, start=1):
                    st.markdown(f"**Turn {idx}:**")
                    st.caption(f"Q: {turn['question']}")
                    st.caption(f"A: {turn['answer'][:120]}...")
            else:
                st.caption("No previous exchanges in memory.")

    # --- Main Chat Area ---
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("📚 Sources & Citations", expanded=False):
                    for src in message["sources"]:
                        page_text = (
                            f", Page {src['page_number']}"
                            if src.get("page_number") is not None
                            else ""
                        )
                        chunk_text = (
                            f" (chunk {src['chunk_index']})"
                            if src.get("chunk_index") is not None
                            else ""
                        )
                        st.markdown(
                            f"- **{src.get('source_name', 'Unknown')}**{page_text}{chunk_text}"
                        )

            if message.get("standalone_question"):
                standalone = message["standalone_question"]
                user_q = message.get("user_question", "")
                if standalone.strip().casefold() != user_q.strip().casefold():
                    st.caption(f"🔍 *Contextual search query:* `{standalone}`")

    # Chat Input
    prompt = st.chat_input("Ask a question about your uploaded documents...")
    if prompt:
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return

        # Display user question
        st.session_state.messages.append(
            {
                "role": "user",
                "content": clean_prompt,
                "user_question": clean_prompt,
            }
        )
        with st.chat_message("user"):
            st.markdown(clean_prompt)

        # Call backend
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating grounded answer..."):
                response = _send_chat_message(
                    session_id=st.session_state.session_id,
                    question=clean_prompt,
                    source_name=selected_source
                    if selected_source != "All Documents"
                    else None,
                )

            if response:
                answer = response.get("answer", "No answer received.")
                sources = response.get("sources", [])
                standalone_question = response.get("standalone_question", clean_prompt)

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources & Citations", expanded=False):
                        for src in sources:
                            page_text = (
                                f", Page {src['page_number']}"
                                if src.get("page_number") is not None
                                else ""
                            )
                            chunk_text = (
                                f" (chunk {src['chunk_index']})"
                                if src.get("chunk_index") is not None
                                else ""
                            )
                            st.markdown(
                                f"- **{src.get('source_name', 'Unknown')}**{page_text}{chunk_text}"
                            )

                if standalone_question.strip().casefold() != clean_prompt.casefold():
                    st.caption(f"🔍 *Contextual search query:* `{standalone_question}`")

                # Store assistant response in history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "standalone_question": standalone_question,
                        "user_question": clean_prompt,
                    }
                )


if __name__ == "__main__":
    main()
