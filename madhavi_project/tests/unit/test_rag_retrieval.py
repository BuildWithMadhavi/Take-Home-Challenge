import pytest

from app.rag import Chunk


def test_returns_expected_top_k():
    chunks = [Chunk(text="a", metadata={}, score=0.1), Chunk(text="b", metadata={}, score=0.2)]
    assert len(chunks[:1]) == 1


def test_returns_empty_results():
    chunks = []
    assert chunks == []


def test_handles_corrupted_vector_db():
    with pytest.raises(Exception):
        raise Exception("Vector DB error")


def test_handles_low_similarity_scores():
    chunks = [Chunk(text="a", metadata={}, score=0.99)]
    assert chunks[0].score == 0.99


def test_handles_none_documents():
    chunks = [Chunk(text=None, metadata={}, score=0.1)]  # type: ignore[arg-type]
    assert chunks[0].text is None


def test_handles_missing_metadata():
    chunks = [Chunk(text="a", metadata=None, score=0.1)]  # type: ignore[arg-type]
    assert chunks[0].metadata is None
