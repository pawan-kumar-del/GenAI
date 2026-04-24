from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from langchain_chroma import Chroma

from rag_support_assistant.config import AppConfig
from rag_support_assistant.ingestion import load_pdf_documents, split_documents_into_chunks
from rag_support_assistant.vector_store import (
    build_embedding_model,
    build_vector_store,
    ingest_documents_into_vector_store,
)
from rag_support_assistant.workflow import build_support_assistant_graph

logger = logging.getLogger(__name__)


def ingest_knowledge_base(config: AppConfig, pdf_path: Path) -> int:
    """Run ingestion pipeline and return number of stored chunks."""
    logger.info("Loading PDF from %s", pdf_path)
    loaded_documents = load_pdf_documents(pdf_path)
    logger.info("Loaded %s document pages", len(loaded_documents))

    chunked_documents = split_documents_into_chunks(
        documents=loaded_documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    logger.info("Generated %s text chunks", len(chunked_documents))

    ingest_documents_into_vector_store(config=config, chunked_documents=chunked_documents)
    logger.info("Ingestion complete. Chunks persisted in ChromaDB.")
    return len(chunked_documents)


def get_vector_store(config: AppConfig) -> Chroma:
    embedding_model = build_embedding_model(config)
    return build_vector_store(config, embedding_model)


def run_support_query(config: AppConfig, user_query: str) -> Dict[str, Any]:
    """Execute LangGraph workflow for a single user query."""
    vector_store = get_vector_store(config)
    support_graph = build_support_assistant_graph(config=config, vector_store=vector_store)
    initial_state: Dict[str, Any] = {
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
    return support_graph.invoke(initial_state)

