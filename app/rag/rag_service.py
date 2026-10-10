
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.rag.chunker import create_document_chunks
from app.rag.document_loader import extract_text_from_pdf
from app.rag.embeddings import LocalEmbeddingService
from app.rag.generation import ClaudeGenerationService
from app.rag.vector_store import VectorStore


class RAGService:
    """Coordinate document indexing, retrieval, and answer generation."""

    def __init__(
        self,
        document_path: Optional[str] = None,
        top_k: int = 3,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        project_root = Path(__file__).resolve().parents[2]

        if document_path is None:
            document_path = str(
                project_root / "documents" / "return_policy.pdf"
            )

        self.document_path = Path(document_path)
        self.top_k = top_k

        self.embedding_service = LocalEmbeddingService()
        self.generation_service = ClaudeGenerationService()

        self.vector_store: Optional[VectorStore] = None

        self._index_documents(chunk_size, chunk_overlap)

    def _index_documents(
        self,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Load, split, embed, and index the document."""

        pages = extract_text_from_pdf(str(self.document_path))

        if not any(page["text"].strip() for page in pages):
            raise ValueError(
                f"No extractable text found in {self.document_path}"
            )

        chunks = create_document_chunks(
            pages=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise ValueError("No document chunks were created")

        embeddings = [
            self.embedding_service.embed_text(chunk["text"])
            for chunk in chunks
        ]

        dimension = len(embeddings[0])

        self.vector_store = VectorStore(dimension=dimension)
        self.vector_store.add(embeddings, chunks)

        print(
            f"Indexed {len(chunks)} chunks "
            f"from {self.document_path.name}"
        )

    def ask(self, question: str) -> Dict[str, Any]:
        """Retrieve relevant chunks and generate an answer."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if self.vector_store is None:
            raise RuntimeError("The vector store has not been initialized")

        query_embedding = self.embedding_service.embed_text(question)

        retrieved_chunks = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        answer = self.generation_service.generate_answer(
            question=question,
            context_chunks=retrieved_chunks,
        )

        sources = [
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "score": chunk["score"],
            }
            for chunk in retrieved_chunks
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }
