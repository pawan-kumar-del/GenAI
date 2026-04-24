from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rag_support_assistant.config import AppConfig


def build_embedding_model(config: AppConfig) -> HuggingFaceEmbeddings:
    """Create the embedding model used for chunk vectorization."""      
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        encode_kwargs={'normalize_embeddings': True}
    )


def build_vector_store(config: AppConfig, embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Return a persistent Chroma vector store instance."""
    return Chroma(
        collection_name=config.chroma_collection_name,
        embedding_function=embedding_model,
        persist_directory=str(config.chroma_persist_directory),
        collection_metadata={"hnsw:space": "cosine"}
    )


def ingest_documents_into_vector_store(
    config: AppConfig, chunked_documents: List[Document]
) -> Chroma:
    """Insert chunked documents into Chroma and persist."""
    embedding_model = build_embedding_model(config)
    vector_store = build_vector_store(config, embedding_model)
    vector_store.add_documents(chunked_documents)
    return vector_store
