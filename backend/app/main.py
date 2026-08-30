from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Settings
from .data import get_call, load_calls
from .evaluation import run_evaluation
from .ingestion import normalise_transcript
from .jobs import JobStore
from .rag import ComplianceRAGWorkflow, analyse_checklist_item, build_chunks

app = FastAPI(title="ClearCall Compliance API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
settings = Settings.from_env()
job_store = JobStore(settings.database_path)


class TranscriptUtterance(BaseModel):
    start_seconds: int = Field(ge=0)
    end_seconds: int = Field(ge=0)
    speaker: str = "Unknown"
    text: str


class ChecklistQuestion(BaseModel):
    id: str
    question: str


class TranscriptIngestionRequest(BaseModel):
    """Transcript input after Azure AI Speech (or another STT service) completes."""

    call_id: str
    title: str = "Uploaded call"
    business_area: str = "Unclassified"
    risk_level: str = "medium"
    utterances: list[TranscriptUtterance]
    checklist: list[ChecklistQuestion]


def process_transcript_job(job_id: str, payload: dict[str, Any]) -> None:
    """Local queue worker: normalise -> retrieve -> generate -> persist results."""
    job_store.update(job_id, "running")
    try:
        utterances = normalise_transcript(payload["utterances"])
        chunks = build_chunks(utterances)
        workflow = ComplianceRAGWorkflow()
        checklist = [item for item in payload["checklist"]]
        results = [workflow.analyse(item, chunks) for item in checklist]
        job_store.update(job_id, "completed", {
            "call_id": payload["call_id"],
            "transcript": utterances,
            "chunks": chunks,
            "results": results,
            "pipeline": {
                "stages": ["transcript_normalisation", "retrieval", "relevance_check", "prompt_augmentation", "answer_generation"],
                "retrieval_provider": workflow.settings.retrieval_provider,
                "generation_provider": workflow.settings.llm_provider,
            },
        })
    except Exception as exc:  # A queued job must record failure rather than lose it.
        job_store.update(job_id, "failed", error=str(exc))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "clearcall-compliance-api",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.ollama_model if settings.llm_provider == "ollama" else settings.azure_openai_chat_deployment,
        "retrieval_provider": settings.retrieval_provider,
        "job_store": "sqlite-local (replace with PostgreSQL in deployment)",
    }


@app.get("/api/calls")
def list_calls() -> list[dict]:
    return [{key: call[key] for key in ("call_id", "title", "business_area", "duration_seconds", "risk_level")} for call in load_calls()]


@app.get("/api/calls/{call_id}")
def call_detail(call_id: str) -> dict:
    try:
        return get_call(call_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Call not found") from exc


@app.post("/api/calls/{call_id}/analyse")
def analyse_call(call_id: str) -> dict:
    try:
        call = get_call(call_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Call not found") from exc
    chunks = build_chunks(call["utterances"])
    workflow = ComplianceRAGWorkflow()
    results = [analyse_checklist_item(item, chunks, workflow) for item in call["checklist"]]
    return {"call_id": call_id, "chunks": chunks, "results": results}


@app.post("/api/ingestions/transcripts", status_code=202)
def ingest_transcript(request: TranscriptIngestionRequest, background_tasks: BackgroundTasks) -> dict:
    """Queue a normalised transcript for asynchronous RAG analysis.

    The production counterpart receives Azure AI Speech output after audio is
    stored securely; this local endpoint accepts the resulting transcript.
    """
    payload = request.model_dump()
    job_id = job_store.create("transcript", payload)
    background_tasks.add_task(process_transcript_job, job_id, payload)
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/jobs/{job_id}"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/dashboard")
def dashboard() -> dict:
    calls = load_calls()
    verdicts = [item["expected_verdict"] for call in calls for item in call["checklist"]]
    return {
        "calls_received": len(calls),
        "checklist_completion": 100,
        "pass_rate": round(100 * verdicts.count("Pass") / len(verdicts), 1),
        "outstanding_items": verdicts.count("Fail") + verdicts.count("Unclear"),
        "high_risk_calls": [call["call_id"] for call in calls if call["risk_level"] == "high"],
    }


@app.post("/api/evaluations/run")
def evaluate() -> dict:
    return run_evaluation(log_to_mlflow=True)
