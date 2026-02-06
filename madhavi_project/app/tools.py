from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DATA_PATH = Path("data/patients.json")


def load_patients() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        return {}
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {p.get("patient_id"): p for p in data}


def get_patient(patient_id: str) -> Dict[str, Any]:
    patients = load_patients()
    return patients.get(patient_id, {})
