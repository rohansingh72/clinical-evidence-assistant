import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Convert text into normalized numerical vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Load the model only when it is first needed."""

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create one normalized embedding for each text."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Create a normalized embedding for a search query."""

        if not query.strip():
            raise ValueError("The search query cannot be empty.")

        return self.embed_texts([query])[0]