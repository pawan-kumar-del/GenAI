from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def normalize_text(raw_text: str) -> str:
    """Normalize whitespace and remove noisy repeated symbols."""
    collapsed_whitespace = re.sub(r"\s+", " ", raw_text).strip()
    cleaned_text = re.sub(r"[ \t]{2,}", " ", collapsed_whitespace)
    return cleaned_text


def load_pdf_documents(pdf_path: Path) -> List[Document]:
    """Load pages from a PDF into LangChain Documents."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file was not found at: {pdf_path}")

    pdf_loader = PyPDFLoader(str(pdf_path))
    loaded_documents = pdf_loader.load()
    for document in loaded_documents:
        document.page_content = normalize_text(document.page_content)

    return loaded_documents


def split_documents_into_chunks(
    documents: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split loaded documents into smaller chunks for embedding and retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    chunked_documents = text_splitter.split_documents(documents)

    for chunk_index, chunk_document in enumerate(chunked_documents):
        chunk_document.metadata["chunk_index"] = chunk_index

    return chunked_documents

