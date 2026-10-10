
from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert text into an embedding vector."""
        pass


class LocalEmbeddingService(EmbeddingService):
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Convert text into a normalized embedding vector."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()
