
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.rag.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application services during startup."""

    app.state.rag_service = RAGService()

    yield

    app.state.rag_service = None


app = FastAPI(
    title="AI Document Assistant API",
    description=(
        "Ask questions about documents using "
        "retrieval-augmented generation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
