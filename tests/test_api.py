
import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeRAGService:
    """Simulate the RAG service without calling external AI models."""

    def ask(self, question: str) -> dict:
        return {
            "question": question,
            "answer": "Customers have 14 days to return a product.",
            "sources": [
                {
                    "source": "return_policy.pdf",
                    "page": 2,
                    "chunk": 1,
                    "score": 0.85,
                }
            ],
        }


class FailingRAGService:
    """Simulate an unexpected RAG service failure."""

    def ask(self, question: str) -> dict:
        raise RuntimeError("Simulated service failure")


@pytest.fixture
def client():
    """Create a test client without starting real AI services."""
    app.state.rag_service = FakeRAGService()
    test_client = TestClient(app)
    yield test_client
    app.state.rag_service = None


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize(
    "question",
    [
        "How many days do customers have to return a product?",
        "¿Cuántos días tienen los clientes para devolver un producto?",
    ],
)
def test_ask_in_english_and_spanish(client, question):
    response = client.post("/ask", json={"question": question})

    assert response.status_code == 200
    assert response.json()["question"] == question
    assert response.json()["answer"]
    assert len(response.json()["sources"]) == 1


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": "   "})

    assert response.status_code == 422


def test_ask_rejects_missing_question(client):
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_returns_503_when_service_is_unavailable(client):
    app.state.rag_service = None

    response = client.post(
        "/ask",
        json={"question": "What is the return policy?"},
    )

    assert response.status_code == 503


def test_ask_returns_500_when_service_fails(client):
    app.state.rag_service = FailingRAGService()

    response = client.post(
        "/ask",
        json={"question": "What is the return policy?"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "An error occurred while processing the question."
    )
