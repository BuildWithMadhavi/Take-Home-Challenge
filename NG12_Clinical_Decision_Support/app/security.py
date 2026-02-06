from __future__ import annotations

from typing import List


INJECTION_PATTERNS: List[str] = [
    "ignore previous",
    "system prompt",
    "developer message",
    "jailbreak",
    "override",
    "bypass",
    "reveal hidden",
    "confidential",
]


def is_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in INJECTION_PATTERNS)


def sanitize_for_logging(text: str) -> str:
    return "[REDACTED]" if text else ""
