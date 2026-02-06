from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RetryPolicy:
    max_attempts: int = 2
    backoff_s: float = 0.5


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_after_s: int = 30) -> None:
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.time() - self._opened_at) >= self.reset_after_s:
            self._opened_at = None
            self._failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.time()


class ResponseCache:
    def __init__(self, max_items: int = 128, ttl_s: int = 600) -> None:
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        if time.time() - item["ts"] > self.ttl_s:
            self._store.pop(key, None)
            return None
        return item["value"]

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_items:
            # Remove oldest item
            oldest_key = min(self._store, key=lambda k: self._store[k]["ts"])
            self._store.pop(oldest_key, None)
        self._store[key] = {"value": value, "ts": time.time()}
