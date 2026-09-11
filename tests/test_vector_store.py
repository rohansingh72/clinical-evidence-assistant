import numpy as np
import pytest

from app.services.text_chunker import DocumentChunk
from app.services.vector_store import InMemoryVectorStore


def create_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        page_number=1,
        chunk_index=1,
        text=text,
    )


def test_vector_store_returns_most_similar_chunk() -> None:
    store = InMemoryVectorStore()

    chunks = [
        create_chunk(
            "safety",
            "The study evaluates adverse events and treatment safety.",
        ),
        create_chunk(
            "dosage",
            "Participants received a 10 mg oral dose.",
        ),
        create_chunk(
            "eligibility",
            "Eligible participants were between 18 and 65 years old.",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, embeddings)

    query_embedding = np.array(
        [0.9, 0.1, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query_embedding,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "safety"
    assert results[0].score > results[1].score


def test_vector_store_rejects_mismatched_counts() -> None:
    store = InMemoryVectorStore()

    chunks = [
        create_chunk(
            "chunk-1",
            "Example text.",
        )
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="number of chunks",
    ):
        store.add(chunks, embeddings)


def test_empty_vector_store_returns_no_results() -> None:
    store = InMemoryVectorStore()

    results = store.search(
        np.array([1.0, 0.0]),
        top_k=3,
    )

    assert results == []