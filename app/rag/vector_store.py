from typing import List, Dict

import faiss
import numpy as np


class VectorStore:
    """Simple FAISS vector store."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Dimension must be greater than 0")

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(dimension)

        self.documents: List[Dict] = []

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[Dict],
    ) -> None:
        """
        Add embeddings and their metadata to the vector store.
        """

        if len(embeddings) != len(documents):
            raise ValueError(
                "Number of embeddings must match "
                "number of documents"
            )

        if not embeddings:
            return

        vectors = np.array(
            embeddings,
            dtype="float32",
        )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

        self.documents.extend(documents)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Search for the most similar documents.
        """

        if not self.documents:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        query_vector = np.array(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Expected dimension {self.dimension}, "
                f"got {query_vector.shape[1]}"
            )

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.documents)),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            document = self.documents[index].copy()

            document["score"] = float(score)

            results.append(document)

        return results