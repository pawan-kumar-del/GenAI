from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma


@dataclass
class RetrievalResult:
    documents: List[Document]
    similarity_scores: List[float]

    @property
    def average_similarity(self) -> float:
        if not self.similarity_scores:
            return 0.0
        return max(self.similarity_scores)


def retrieve_relevant_documents(
    vector_store: Chroma, query: str, top_k: int
) -> RetrievalResult:
    """Retrieve top-k most relevant chunks with relevance scores."""
    if not query.strip():
        return RetrievalResult(documents=[], similarity_scores=[])

    retrieved_pairs = vector_store.similarity_search_with_relevance_scores(
        query=query, k=top_k
    )
    retrieved_documents = [document for document, _ in retrieved_pairs]
    relevance_scores = [float(score) for _, score in retrieved_pairs]
    return RetrievalResult(documents=retrieved_documents, similarity_scores=relevance_scores)


def is_query_ambiguous(cleaned_query: str) -> bool:
    """Heuristic ambiguity detection for HITL routing."""
    lower_query = cleaned_query.lower().strip()
    ambiguity_markers = [
        "help",
        "issue",
        "problem",
        "not working",
        "what should i do",
        "can you assist",
    ]
    is_too_short = len(lower_query.split()) < 4
    contains_vague_phrase = any(marker == lower_query for marker in ambiguity_markers)
    return is_too_short or contains_vague_phrase

