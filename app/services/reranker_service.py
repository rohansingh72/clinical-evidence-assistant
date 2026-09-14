from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


class RerankerService:
    """Score passages by their relevance to a question."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)

        return self._model

    def score_passages(
        self,
        question: str,
        passages: list[str],
    ) -> np.ndarray:
        if not question.strip():
            raise ValueError("The question cannot be empty.")

        if not passages:
            return np.array([], dtype=np.float32)

        pairs = [
            [question, passage]
            for passage in passages
        ]

        scores = self.model.predict(pairs)

        return np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)