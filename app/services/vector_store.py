from dataclasses import dataclass

import numpy as np

from app.services.text_chunker import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


class InMemoryVectorStore:
    """
    Store document chunks and search them using cosine similarity.

    Embeddings are normalized before storage, so their dot product
    represents cosine similarity.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        if not chunks:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError("Embeddings cannot contain zero vectors.")

        normalized_embeddings = embeddings / norms

        if self._embeddings is None:
            self._embeddings = normalized_embeddings
        else:
            if (
                normalized_embeddings.shape[1]
                != self._embeddings.shape[1]
            ):
                raise ValueError(
                    "New embeddings have an incompatible dimension."
                )

            self._embeddings = np.vstack(
                [self._embeddings, normalized_embeddings]
            )

        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[SearchResult]:
        if self._embeddings is None or not self._chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        ).reshape(-1)

        if query_embedding.shape[0] != self._embeddings.shape[1]:
            raise ValueError(
                "The query embedding has an incompatible dimension."
            )

        query_norm = np.linalg.norm(query_embedding)

        if query_norm == 0:
            raise ValueError("The query embedding cannot be a zero vector.")

        normalized_query = query_embedding / query_norm
        similarity_scores = self._embeddings @ normalized_query

        result_count = min(top_k, len(self._chunks))
        result_indices = np.argsort(-similarity_scores)[:result_count]

        return [
            SearchResult(
                chunk=self._chunks[index],
                score=float(similarity_scores[index]),
            )
            for index in result_indices
        ]