from __future__ import annotations

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .rag import CodeRAGService

load_dotenv(find_dotenv())

app = FastAPI(title="LangChain Code RAG Backend", version="0.1.0")
service = CodeRAGService()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_dir: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> dict[str, int | str]:
    return service.ingest()


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = service.answer(request.question, top_k=request.top_k)
    return AskResponse(answer=result.answer, sources=result.sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("langchain_backend.main:app", host="0.0.0.0", port=8001, reload=False)
