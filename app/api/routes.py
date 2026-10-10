
import os
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


def verify_api_key(
    x_api_key: str = Header(default="", alias="X-API-Key"),
):
    """Validate the API key supplied by the client."""

    expected_key = os.getenv("APP_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if not x_api_key or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


class QuestionRequest(BaseModel):
    """Request payload for document questions."""

    question: str = Field(min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    """Metadata for a retrieved document chunk."""

    source: str
    page: int
    chunk: int
    score: float


class QuestionResponse(BaseModel):
    """Response payload containing the answer and its sources."""

    question: str
    answer: str
    sources: list[SourceResponse]


@router.get("/health")
def health_check():
    """Check whether the API is running."""

    return {"status": "healthy"}


@router.post(
    "/ask",
    response_model=QuestionResponse,
    dependencies=[Depends(verify_api_key)],
)
def ask_question(
    payload: QuestionRequest,
    request: Request,
):
    """Answer a question using the indexed documents."""

    if not payload.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    rag_service = getattr(
        request.app.state,
        "rag_service",
        None,
    )

    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG service is not available.",
        )

    try:
        return rag_service.ask(payload.question.strip())
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        )
