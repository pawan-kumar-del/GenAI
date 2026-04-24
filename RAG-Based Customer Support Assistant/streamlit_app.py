from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st

from rag_support_assistant.config import AppConfig, get_config
from rag_support_assistant.hitl_store import create_ticket, list_tickets, resolve_ticket
from rag_support_assistant.logging_utils import configure_logging
from rag_support_assistant.service import get_vector_store, ingest_knowledge_base
from rag_support_assistant.workflow import build_support_assistant_graph


def build_initial_state(user_query: str) -> Dict[str, Any]:
    return {
        "query": user_query,
        "cleaned_query": "",
        "retrieved_documents": [],
        "similarity_scores": [],
        "answer": "",
        "confidence": 0.0,
        "needs_escalation": False,
        "escalation_reason": "",
        "error": "",
    }


def ensure_session_state() -> None:
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "delivered_human_ticket_ids" not in st.session_state:
        st.session_state.delivered_human_ticket_ids = []


@st.cache_resource(show_spinner=False)
def load_runtime() -> tuple[AppConfig, Any]:
    config = get_config()
    configure_logging(config.log_level)
    vector_store = get_vector_store(config)
    graph = build_support_assistant_graph(config=config, vector_store=vector_store)
    return config, graph


def summarize_sources(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    documents = final_state.get("retrieved_documents", [])
    scores = final_state.get("similarity_scores", [])
    sources: List[Dict[str, Any]] = []

    for index, document in enumerate(documents):
        metadata = getattr(document, "metadata", {}) or {}
        source_path = metadata.get("source", "Unknown source")
        page_number = metadata.get("page")
        score = scores[index] if index < len(scores) else None
        excerpt = getattr(document, "page_content", "").strip()
        sources.append(
            {
                "label": f"Source {index + 1}",
                "source": Path(source_path).name if source_path else "Unknown source",
                "page": page_number + 1 if isinstance(page_number, int) else None,
                "score": score,
                "excerpt": excerpt[:800] + ("..." if len(excerpt) > 800 else ""),
            }
        )

    return sources


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander("Retrieved context", expanded=False):
        for source in sources:
            label_parts = [source["label"], source["source"]]
            if source["page"] is not None:
                label_parts.append(f"page {source['page']}")
            if source["score"] is not None:
                label_parts.append(f"score {source['score']:.2f}")

            st.markdown(f"**{' | '.join(label_parts)}**")
            st.caption(source["excerpt"])


def save_uploaded_pdf(uploaded_file: Any, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / Path(uploaded_file.name).name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def sync_human_responses() -> None:
    session_id = st.session_state.chat_session_id
    delivered_ticket_ids = set(st.session_state.delivered_human_ticket_ids)
    resolved_tickets = list_tickets(status="resolved", session_id=session_id)

    for ticket in resolved_tickets:
        ticket_id = ticket["ticket_id"]
        if ticket_id in delivered_ticket_ids:
            continue

        response = ticket.get("human_response", "").strip()
        if not response:
            continue

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Human support response for ticket `{ticket_id}`:\n\n{response}"
                ),
                "confidence": 1.0,
                "needs_escalation": False,
                "sources": ticket.get("retrieved_context", []),
                "response_source": "human_agent",
            }
        )
        delivered_ticket_ids.add(ticket_id)

    st.session_state.delivered_human_ticket_ids = sorted(delivered_ticket_ids)


def render_customer_sidebar(config: AppConfig) -> None:
    st.header("System")
    st.write(f"Collection: `{config.chroma_collection_name}`")
    st.write(f"Vector DB: `{config.chroma_persist_directory}`")
    st.write(f"Knowledge base: `{config.default_pdf_path}`")
    st.write(f"Top K: `{config.retrieval_top_k}`")
    st.write(f"Threshold: `{config.similarity_threshold:.2f}`")

    if Path(config.chroma_persist_directory).exists():
        st.success("Chroma DB found")
    else:
        st.warning("Chroma DB not found. Run ingestion first.")

    st.divider()
    st.subheader("Upload and ingest PDF")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF knowledge base",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload a PDF and ingest it into the live Chroma database.",
    )

    if uploaded_pdf is not None:
        st.caption(f"Selected file: `{uploaded_pdf.name}`")

    if st.button("Ingest uploaded PDF", use_container_width=True):
        if uploaded_pdf is None:
            st.warning("Upload a PDF before starting ingestion.")
        else:
            try:
                upload_path = save_uploaded_pdf(uploaded_pdf, Path("data/uploads"))
                with st.spinner("Ingesting uploaded PDF into ChromaDB..."):
                    chunk_count = ingest_knowledge_base(config=config, pdf_path=upload_path)
            except Exception as error:
                st.error(f"PDF ingestion failed: {error}")
            else:
                load_runtime.clear()
                st.session_state.messages = []
                st.session_state.last_ingested_pdf = str(upload_path)
                st.success(
                    f"Ingestion complete for `{upload_path.name}`. Stored {chunk_count} chunks."
                )
                st.rerun()

    last_ingested_pdf = st.session_state.get("last_ingested_pdf")
    if last_ingested_pdf:
        st.caption(f"Last ingested from app: `{Path(last_ingested_pdf).name}`")

    pending_ticket_count = len(
        list_tickets(status="pending", session_id=st.session_state.chat_session_id)
    )
    st.caption(f"Open escalations for this chat: {pending_ticket_count}")

    if st.button("Check for human replies", use_container_width=True):
        sync_human_responses()
        st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.delivered_human_ticket_ids = []
        st.rerun()


def render_agent_view() -> None:
    st.subheader("Agent Console")
    st.caption("Review pending escalations and send the human response back into the chat.")

    pending_tickets = list_tickets(status="pending")
    if not pending_tickets:
        st.info("No pending escalation tickets.")
        return

    agent_name = st.text_input("Agent name", value="Support Agent")
    for ticket in pending_tickets:
        ticket_id = ticket["ticket_id"]
        title = f"{ticket_id} | confidence={ticket['confidence']:.2f}"
        with st.expander(title, expanded=False):
            st.markdown(f"**Customer question**\n\n{ticket['user_query']}")
            st.markdown(f"**Escalation reason**\n\n{ticket['escalation_reason']}")
            st.caption(
                f"Session: `{ticket['session_id']}` | Created: `{ticket['created_at']}`"
            )
            render_sources(ticket.get("retrieved_context", []))
            response_key = f"agent_response_{ticket_id}"
            human_response = st.text_area(
                "Human response",
                key=response_key,
                height=180,
                placeholder="Write the answer that should be returned to the customer.",
            )
            if st.button("Submit response", key=f"resolve_{ticket_id}"):
                if not human_response.strip():
                    st.warning("Enter a response before resolving the ticket.")
                else:
                    resolve_ticket(
                        ticket_id=ticket_id,
                        human_response=human_response.strip(),
                        resolved_by=agent_name.strip() or "Support Agent",
                    )
                    st.success(f"Resolved {ticket_id}.")
                    st.rerun()


def render_sidebar(config: AppConfig) -> str:
    with st.sidebar:
        st.header("Mode")
        selected_view = st.radio(
            "Select interface",
            options=["Customer View", "Agent View"],
            index=0,
        )
        st.divider()

        if selected_view == "Customer View":
            render_customer_sidebar(config)
        else:
            st.caption("Use the main panel to review and resolve escalation tickets.")
            if st.button("Refresh tickets", use_container_width=True):
                st.rerun()

        return selected_view


def main() -> None:
    st.set_page_config(
        page_title="Customer Support Assistant",
        page_icon="💬",
        layout="wide",
    )
    st.title("Customer Support Assistant")
    st.caption("RAG support assistant with retrieval confidence and escalation flow.")
    ensure_session_state()

    try:
        config, support_graph = load_runtime()
    except Exception as error:
        st.error(f"App failed to initialize: {error}")
        st.info("Check `.env`, ensure `GROQ_API_KEY` is set, and confirm the vector store exists.")
        return

    selected_view = render_sidebar(config)
    if selected_view == "Agent View":
        render_agent_view()
        return

    sync_human_responses()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                response_source = message.get("response_source", "rag")
                st.caption(
                    f"source={response_source} | confidence={message['confidence']:.2f} | "
                    f"escalated={str(message['needs_escalation']).lower()}"
                )
                render_sources(message.get("sources", []))

    user_query = st.chat_input("Ask a support question")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            final_state = support_graph.invoke(build_initial_state(user_query))

        answer = final_state.get("answer", "No answer returned.")
        confidence = float(final_state.get("confidence", 0.0))
        needs_escalation = bool(final_state.get("needs_escalation", False))
        sources = summarize_sources(final_state)
        response_source = "rag"

        if needs_escalation:
            escalation_reason = final_state.get(
                "escalation_reason", "Escalated by workflow policy."
            )
            ticket = create_ticket(
                session_id=st.session_state.chat_session_id,
                user_query=user_query,
                escalation_reason=escalation_reason,
                confidence=confidence,
                retrieved_context=sources,
            )
            answer = (
                f"{answer}\n\n"
                f"A human support ticket has been created: `{ticket['ticket_id']}`.\n\n"
                "Switch to `Agent View` to answer it, or stay here and click "
                "`Check for human replies` when the agent has responded."
            )
            response_source = "hitl_escalation"

        st.markdown(answer)
        st.caption(
            f"source={response_source} | confidence={confidence:.2f} | "
            f"escalated={str(needs_escalation).lower()}"
        )
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "sources": sources,
            "response_source": response_source,
        }
    )


if __name__ == "__main__":
    main()
