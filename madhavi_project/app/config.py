from __future__ import annotations

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CDS_", case_sensitive=False)

    project_id: str = Field(..., description="GCP project id")
    location: str = Field("us-central1", description="GCP region")

    gemini_model: str = Field("gemini-1.5-pro", description="Vertex AI Gemini model name")
    embedding_model: str = Field("text-embedding-004", description="Vertex AI embedding model name")

    chroma_path: str = Field("data/chroma", description="ChromaDB persistent path")
    collection_name: str = Field("ng12", description="ChromaDB collection name")

    top_k: int = Field(5, ge=1, le=20)
    min_similarity: Optional[float] = Field(None, description="Optional similarity threshold")

    log_level: str = Field("INFO")
    read_only: bool = Field(True, description="Prevent writes to vector DB at runtime")

    request_timeout_s: int = Field(30, ge=1, le=120)
    max_history_turns: int = Field(6, ge=0, le=20)

    retry_max_attempts: int = Field(2, ge=1, le=5)
    retry_backoff_s: float = Field(0.5, ge=0.1, le=5.0)
    breaker_threshold: int = Field(3, ge=1, le=10)
    breaker_reset_s: int = Field(30, ge=5, le=300)
    cache_ttl_s: int = Field(600, ge=60, le=3600)
    cache_max_items: int = Field(128, ge=10, le=1000)


settings = Settings()
