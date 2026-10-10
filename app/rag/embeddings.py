
import os
from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingService(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Convert text into an embedding vector."""
        raise NotImplementedError


class LocalEmbeddingService(EmbeddingService):
    """Generate embeddings locally using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        model_path = os.getenv("EMBEDDING_MODEL_PATH", model_name)

        self.model = SentenceTransformer(
            model_path,
            local_files_only=os.getenv("HF_HUB_OFFLINE") == "1",
        )

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
