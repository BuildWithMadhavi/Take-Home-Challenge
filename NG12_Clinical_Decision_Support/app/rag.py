from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import chromadb
import vertexai
from vertexai.language_models import TextEmbeddingModel

from .config import settings
from .resilience import CircuitBreaker, RetryPolicy


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    score: float


class VertexEmbeddingClient:
    def __init__(self) -> None:
        vertexai.init(project=settings.project_id, location=settings.location)
        self._model = TextEmbeddingModel.from_pretrained(settings.embedding_model)

    def embed(self, text: str) -> List[float]:
        embeddings = self._model.get_embeddings([text])
        return embeddings[0].values


class ChromaRAG:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.chroma_path)
        self._collection = self._client.get_collection(settings.collection_name)
        self._embedder = VertexEmbeddingClient()
        self._breaker = CircuitBreaker(
            failure_threshold=settings.breaker_threshold,
            reset_after_s=settings.breaker_reset_s,
        )
        self._retry = RetryPolicy(
            max_attempts=settings.retry_max_attempts,
            backoff_s=settings.retry_backoff_s,
        )

    def query(self, question: str, top_k: int) -> List[Chunk]:
        if self._breaker.is_open():
            return []

        attempt = 0
        last_err: Exception | None = None
        while attempt < self._retry.max_attempts:
            attempt += 1
            try:
                embedding = self._embedder.embed(question)
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                self._breaker.record_success()
                break
            except Exception as exc:
                last_err = exc
                self._breaker.record_failure()
                if attempt < self._retry.max_attempts:
                    import time
                    time.sleep(self._retry.backoff_s)
        else:
            if last_err:
                raise last_err

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: List[Chunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if doc is None or meta is None:
                continue
            chunks.append(Chunk(text=doc, metadata=meta, score=dist))
        return chunks
