from __future__ import annotations

import json
from typing import Any, Dict, List

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

from .config import settings
from .models import AgentOutput
from .security import is_prompt_injection
from .resilience import CircuitBreaker, RetryPolicy, ResponseCache


DISCLAIMER = "This tool supports clinical decision-making and does not provide diagnoses."


SYSTEM_PROMPT = """
You are a compliance-bound clinical reasoning system.
Rules:
- Use ONLY the provided NG12 context.
- Do NOT infer clinical thresholds.
- If evidence is missing, return Insufficient Evidence.
- ALL clinical statements must be supported by citations from the NG12 context.
- Output must be valid JSON with keys: assessment, reasoning, citations, confidence.
- citations must include source, page, chunk_id, excerpt.
""".strip()

_breaker = CircuitBreaker(
    failure_threshold=settings.breaker_threshold,
    reset_after_s=settings.breaker_reset_s,
)
_retry = RetryPolicy(
    max_attempts=settings.retry_max_attempts,
    backoff_s=settings.retry_backoff_s,
)
_cache = ResponseCache(
    max_items=settings.cache_max_items,
    ttl_s=settings.cache_ttl_s,
)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for ch in chunks:
        meta = ch.get("metadata", {})
        chunk_id = meta.get("chunk_id", "")
        page = meta.get("page", "")
        text = ch.get("text", "")
        lines.append(f"[{chunk_id}|page {page}] {text}")
    return "\n".join(lines)


def _base_failure() -> AgentOutput:
    return AgentOutput(
        assessment="Insufficient Evidence",
        reasoning="Insufficient evidence in retrieved NG12 text.",
        citations=[],
        confidence="low",
    )


def _parse_output(raw: str) -> AgentOutput:
    data = json.loads(raw)
    return AgentOutput(**data)


def _validate_output(output: AgentOutput, chunks: List[Dict[str, Any]]) -> AgentOutput:
    if output.assessment != "Insufficient Evidence" and not output.citations:
        return _base_failure()

    by_chunk = {c.get("metadata", {}).get("chunk_id"): c.get("text", "") for c in chunks}
    for c in output.citations:
        source_text = by_chunk.get(c.chunk_id, "")
        if not source_text or c.excerpt not in source_text:
            return _base_failure()
    return output


def generate_assessment(question: str, chunks: List[Dict[str, Any]]) -> tuple[AgentOutput, Dict[str, str]]:
    meta: Dict[str, str] = {}
    if is_prompt_injection(question):
        meta["reason"] = "prompt_injection"
        return _base_failure(), meta
    if not chunks:
        meta["reason"] = "no_chunks"
        cached = _cache.get(question)
        if cached:
            meta["cache"] = "hit"
            return cached, meta
        return _base_failure(), meta

    if _breaker.is_open():
        meta["reason"] = "breaker_open"
        cached = _cache.get(question)
        if cached:
            meta["cache"] = "hit"
            return cached, meta
        return _base_failure(), meta

    vertexai.init(project=settings.project_id, location=settings.location)
    context = _format_context(chunks)

    prompt = f"""
{SYSTEM_PROMPT}

NG12 context:
{context}

Question:
{question}

Return JSON only.
""".strip()

    try:
        response_text = _generate_with_retries(prompt)
        output = _parse_output(response_text)
        output = _validate_output(output, chunks)
        if output.citations:
            _cache.set(question, output)
        _breaker.record_success()
        return output, meta
    except Exception:
        meta["reason"] = "generation_failed"
        cached = _cache.get(question)
        if cached:
            meta["cache"] = "hit"
            return cached, meta
        return _base_failure(), meta


def _generate_with_retries(prompt: str) -> str:
    attempt = 0
    last_err: Exception | None = None
    while attempt < _retry.max_attempts:
        attempt += 1
        try:
            response = _call_gemini(prompt)
            return response
        except Exception as exc:
            last_err = exc
            _breaker.record_failure()
            if attempt < _retry.max_attempts:
                import time
                time.sleep(_retry.backoff_s)
    if last_err:
        raise last_err
    raise RuntimeError("Unknown generation failure")


def _call_gemini(prompt: str) -> str:
    model = GenerativeModel(settings.gemini_model)
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0.0,
            top_p=0.0,
            top_k=1,
            max_output_tokens=512,
        ),
        request_options={"timeout": settings.request_timeout_s},
    )
    return response.text
