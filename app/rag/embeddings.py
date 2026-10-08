from abc import ABC, abstractmethod
from typing import List


class EmbeddingService(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Convert text into an embedding vector.
        """
        pass