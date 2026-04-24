from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from rag_support_assistant.config import AppConfig


SYSTEM_PROMPT = """You are a customer support assistant using Retrieval-Augmented Generation.
Follow these strict rules:
1) Answer only with information grounded in the provided context.
2) If context is insufficient, do not guess. Say you need escalation to a human support agent.
3) Keep responses concise, actionable, and customer-friendly.
4) Cite relevant policy/process details from context in plain language.
5) Never fabricate product details, pricing, timelines, or legal claims.
"""


def build_llm(config: AppConfig) -> ChatGroq:
    """Create a Groq chat model instance."""
    return ChatGroq(
        api_key=config.groq_api_key,
        model=config.groq_model,
        temperature=0.1,
    )


def generate_grounded_answer(
    llm: ChatGroq, user_query: str, retrieved_documents: List[Document]
) -> str:
    """Generate an answer constrained to retrieved context."""
    if not retrieved_documents:
        return (
            "I could not find relevant information in the knowledge base. "
            "I will escalate this to a human support specialist."
        )

    context = "\n\n".join(
        f"[Chunk {index + 1}] {document.page_content}"
        for index, document in enumerate(retrieved_documents)
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "User question:\n{query}\n\nRetrieved context:\n{context}\n\n"
                "Provide the best grounded answer. If context is not enough, say escalation is required.",
            ),
        ]
    )
    response = llm.invoke(prompt_template.format_messages(query=user_query, context=context))
    return response.content

