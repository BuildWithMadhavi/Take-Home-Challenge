from __future__ import annotations

import json
import uuid
from typing import Dict

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .agent import DISCLAIMER, generate_assessment
from .config import settings
from .health import health_payload
from .memory import trim_history
from .models import AssessRequest, AssessResponse, ChatRequest, ChatResponse, ErrorInfo, AgentOutput
from .rag import ChromaRAG
from .security import sanitize_for_logging
from .tools import get_patient

app = FastAPI(title="Clinical Reasoning Platform", version="1.0")
_rag_instance: ChromaRAG | None = None


def _cid(correlation_id: str | None) -> str:
    return correlation_id or str(uuid.uuid4())


def _log(event: str, data: Dict[str, str]) -> None:
    payload = {"event": event, **data}
    print(json.dumps(payload))


def _get_rag() -> ChromaRAG | None:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance
    try:
        _rag_instance = ChromaRAG()
        return _rag_instance
    except Exception:
        return None


@app.get("/health")
def health() -> Dict[str, str]:
    return health_payload()


def _unwrap_result(value) -> tuple[AgentOutput, dict]:
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1] or {}
    return value, {}


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest, x_correlation_id: str | None = Header(default=None)) -> JSONResponse:
    correlation_id = _cid(x_correlation_id)
    patient = get_patient(req.patient_id)
    if not patient:
        raise HTTPException(status_code=400, detail="Invalid patient_id")

    symptoms = patient.get("symptoms", [])
    if not symptoms:
        raise HTTPException(status_code=400, detail="Empty symptoms list")

    question = f"Assess NG12 risk for symptoms: {', '.join(symptoms)}"

    errors: list[ErrorInfo] = []
    rag = _get_rag()
    try:
        chunks = [
            {"text": c.text, "metadata": c.metadata, "score": c.score}
            for c in (rag.query(question, settings.top_k) if rag else [])
        ]
    except Exception:
        errors.append(ErrorInfo(code="RETRIEVAL_FAILED", message="Vector retrieval failed"))
        chunks = []
    if not chunks:
        errors.append(ErrorInfo(code="RETRIEVAL_EMPTY", message="No retrieval results"))

    try:
        result, meta = _unwrap_result(generate_assessment(question, chunks))
    except Exception:
        errors.append(ErrorInfo(code="GENERATION_FAILED", message="Generation failed"))
        result, meta = _unwrap_result(generate_assessment("", []))

    if meta.get("reason") == "prompt_injection":
        errors.append(ErrorInfo(code="PROMPT_INJECTION", message="Prompt injection detected"))
    if meta.get("reason") == "breaker_open":
        errors.append(ErrorInfo(code="CIRCUIT_OPEN", message="Circuit breaker open"))
    if meta.get("reason") == "generation_failed":
        errors.append(ErrorInfo(code="GENERATION_FAILED", message="Generation failed"))
    if meta.get("cache") == "hit":
        errors.append(ErrorInfo(code="CACHE_HIT", message="Returned cached response"))

    _log(
        "assess",
        {
            "correlation_id": correlation_id,
            "retrieval_count": str(len(chunks)),
            "citation_count": str(len(result.citations)),
            "confidence": result.confidence,
            "patient_id": sanitize_for_logging(req.patient_id),
        },
    )

    status = "ok" if not errors else "degraded"
    response = AssessResponse(
        correlation_id=correlation_id,
        disclaimer=DISCLAIMER,
        result=result,
        status=status,
        errors=errors or None,
    )
    return JSONResponse(content=response.model_dump())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_correlation_id: str | None = Header(default=None)) -> JSONResponse:
    correlation_id = _cid(x_correlation_id)
    history = [m.model_dump() for m in req.history] if req.history else []
    history = trim_history(history, settings.max_history_turns)

    question = req.question
    if history:
        history_block = "\n".join([f"{h['role']}: {h['content']}" for h in history])
        question = (
            "Conversation so far:\n"
            f"{history_block}\n\n"
            f"User question: {req.question}"
        )

    errors: list[ErrorInfo] = []
    rag = _get_rag()
    try:
        chunks = [
            {"text": c.text, "metadata": c.metadata, "score": c.score}
            for c in (rag.query(req.question, settings.top_k) if rag else [])
        ]
    except Exception:
        errors.append(ErrorInfo(code="RETRIEVAL_FAILED", message="Vector retrieval failed"))
        chunks = []
    if not chunks:
        errors.append(ErrorInfo(code="RETRIEVAL_EMPTY", message="No retrieval results"))

    try:
        result, meta = _unwrap_result(generate_assessment(question, chunks))
    except Exception:
        errors.append(ErrorInfo(code="GENERATION_FAILED", message="Generation failed"))
        result, meta = _unwrap_result(generate_assessment("", []))

    if meta.get("reason") == "prompt_injection":
        errors.append(ErrorInfo(code="PROMPT_INJECTION", message="Prompt injection detected"))
    if meta.get("reason") == "breaker_open":
        errors.append(ErrorInfo(code="CIRCUIT_OPEN", message="Circuit breaker open"))
    if meta.get("reason") == "generation_failed":
        errors.append(ErrorInfo(code="GENERATION_FAILED", message="Generation failed"))
    if meta.get("cache") == "hit":
        errors.append(ErrorInfo(code="CACHE_HIT", message="Returned cached response"))

    _log(
        "chat",
        {
            "correlation_id": correlation_id,
            "retrieval_count": str(len(chunks)),
            "citation_count": str(len(result.citations)),
            "confidence": result.confidence,
        },
    )

    status = "ok" if not errors else "degraded"
    response = ChatResponse(
        correlation_id=correlation_id,
        disclaimer=DISCLAIMER,
        result=result,
        status=status,
        errors=errors or None,
    )
    return JSONResponse(content=response.model_dump())
