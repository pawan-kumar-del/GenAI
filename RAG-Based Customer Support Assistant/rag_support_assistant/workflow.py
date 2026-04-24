from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from rag_support_assistant.config import AppConfig
from rag_support_assistant.llm_service import build_llm, generate_grounded_answer
from rag_support_assistant.retriever import is_query_ambiguous, retrieve_relevant_documents


class SupportAssistantState(TypedDict):
    query: str
    cleaned_query: str
    retrieved_documents: List[Document]
    similarity_scores: List[float]
    answer: str
    confidence: float
    needs_escalation: bool
    escalation_reason: str
    error: str


def normalize_user_query(query: str) -> str:
    return " ".join(query.strip().split())


def input_processing_node(state: SupportAssistantState) -> Dict[str, Any]:
    cleaned_query = normalize_user_query(state.get("query", ""))
    if not cleaned_query:
        return {
            "cleaned_query": cleaned_query,
            "error": "Invalid input: query cannot be empty.",
            "needs_escalation": True,
            "escalation_reason": "Invalid input query.",
            "answer": "Please provide a non-empty question so I can help you.",
        }

    return {"cleaned_query": cleaned_query}


def make_retrieve_and_answer_node(config: AppConfig, vector_store: Chroma):
    llm = build_llm(config)

    def retrieve_and_answer_node(state: SupportAssistantState) -> Dict[str, Any]:
        if state.get("error"):
            return {}

        retrieval_result = retrieve_relevant_documents(
            vector_store=vector_store,
            query=state["cleaned_query"],
            top_k=config.retrieval_top_k,
        )
        try:
            answer = generate_grounded_answer(
                llm=llm,
                user_query=state["cleaned_query"],
                retrieved_documents=retrieval_result.documents,
            )
            runtime_error = ""
        except Exception as llm_error:
            answer = (
                "I am unable to complete this response due to a language model service error. "
                "I will escalate this to a human support specialist."
            )
            runtime_error = f"LLM runtime error: {llm_error}"
        return {
            "retrieved_documents": retrieval_result.documents,
            "similarity_scores": retrieval_result.similarity_scores,
            "answer": answer,
            "confidence": retrieval_result.average_similarity,
            "error": runtime_error,
        }

    return retrieve_and_answer_node


def make_decision_node(config: AppConfig):
    def decision_node(state: SupportAssistantState) -> Dict[str, Any]:
        if state.get("error"):
            return {
                "needs_escalation": True,
                "escalation_reason": state["error"],
            }

        documents = state.get("retrieved_documents", [])
        confidence_score = state.get("confidence", 0.0)
        cleaned_query = state.get("cleaned_query", "")

        if not documents:
            return {
                "needs_escalation": True,
                "escalation_reason": "No relevant documents found in knowledge base.",
            }

        if confidence_score < config.similarity_threshold:
            return {
                "needs_escalation": True,
                "escalation_reason": (
                    f"Low confidence retrieval score ({confidence_score:.2f}) "
                    f"below threshold ({config.similarity_threshold:.2f})."
                ),
            }

        if is_query_ambiguous(cleaned_query):
            return {
                "needs_escalation": True,
                "escalation_reason": "Query appears ambiguous and requires human clarification.",
            }

        return {"needs_escalation": False, "escalation_reason": ""}

    return decision_node


def route_after_decision(state: SupportAssistantState) -> Literal["output_node", "escalation_node"]:
    if state.get("needs_escalation", False):
        return "escalation_node"
    return "output_node"


def output_node(state: SupportAssistantState) -> Dict[str, Any]:
    return {"answer": state.get("answer", "No answer available.")}


def escalation_node(state: SupportAssistantState) -> Dict[str, Any]:
    escalation_reason = state.get("escalation_reason", "Escalation requested by workflow.")
    escalation_message = (
        "I am escalating this conversation to a human support specialist because: "
        f"{escalation_reason}"
    )
    return {"answer": escalation_message}


def build_support_assistant_graph(config: AppConfig, vector_store: Chroma):
    graph_builder = StateGraph(SupportAssistantState)

    graph_builder.add_node("input_processing_node", input_processing_node)
    graph_builder.add_node(
        "retrieve_and_answer_node", make_retrieve_and_answer_node(config, vector_store)
    )
    graph_builder.add_node("decision_node", make_decision_node(config))
    graph_builder.add_node("output_node", output_node)
    graph_builder.add_node("escalation_node", escalation_node)

    graph_builder.add_edge(START, "input_processing_node")
    graph_builder.add_edge("input_processing_node", "retrieve_and_answer_node")
    graph_builder.add_edge("retrieve_and_answer_node", "decision_node")
    graph_builder.add_conditional_edges(
        "decision_node",
        route_after_decision,
        {
            "output_node": "output_node",
            "escalation_node": "escalation_node",
        },
    )
    graph_builder.add_edge("output_node", END)
    graph_builder.add_edge("escalation_node", END)

    return graph_builder.compile()

