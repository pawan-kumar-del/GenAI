from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """Application-level configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")

    embedding_model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(4, alias="RETRIEVAL_TOP_K")
    similarity_threshold: float = Field(0.30, alias="SIMILARITY_THRESHOLD")

    chroma_collection_name: str = Field("support_knowledge_base", alias="CHROMA_COLLECTION")
    chroma_persist_directory: Path = Field(
        Path("./chroma_db"), alias="CHROMA_PERSIST_DIRECTORY"
    )
    default_pdf_path: Path = Field(Path("./data/knowledge_base.pdf"), alias="PDF_PATH")

    log_level: str = Field("INFO", alias="LOG_LEVEL")


def get_config() -> AppConfig:
    """Create a validated application configuration instance."""
    return AppConfig()

