import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _use_mock_patients(monkeypatch):
    from app import tools

    test_path = Path("tests/test_data/mock_patients.json")
    monkeypatch.setattr(tools, "DATA_PATH", test_path)
    return None


@pytest.fixture()
def mock_chunks():
    return [
        {
            "text": "NG12 excerpt placeholder text.",
            "metadata": {"source": "NG12 PDF", "page": 1, "chunk_id": "ng12_p1_c0"},
            "score": 0.1,
        }
    ]


@pytest.fixture()
def mock_agent_output():
    return {
        "assessment": "Insufficient Evidence",
        "reasoning": "Insufficient evidence in retrieved NG12 text.",
        "citations": [],
        "confidence": "low",
    }


@pytest.fixture()
def mock_urgent_output():
    return {
        "assessment": "Urgent Referral",
        "reasoning": "Supported by NG12 excerpt.",
        "citations": [
            {"source": "NG12 PDF", "page": 1, "chunk_id": "ng12_p1_c0", "excerpt": "NG12 excerpt"}
        ],
        "confidence": "high",
    }
