from __future__ import annotations

from typing import Dict

from .config import settings


def health_payload() -> Dict[str, str]:
    return {"status": "ok", "collection": settings.collection_name}
