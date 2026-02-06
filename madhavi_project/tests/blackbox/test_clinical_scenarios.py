from fastapi.testclient import TestClient

from app.main import app
from app.models import AgentOutput

client = TestClient(app)


def _setup(monkeypatch, mock_agent_output):
    from app import main

    class _Rag:
        def query(self, q, k):
            return []

    monkeypatch.setattr(main, "_rag_instance", _Rag())
    monkeypatch.setattr(main, "generate_assessment", lambda q, c: AgentOutput(**mock_agent_output))


def test_persistent_hoarseness(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Persistent hoarseness"})
    assert res.status_code == 200


def test_visible_haematuria(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Visible haematuria"})
    assert res.status_code == 200


def test_dyspepsia_under_age_threshold(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Dyspepsia under age threshold"})
    assert res.status_code == 200


def test_fatigue_only(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Fatigue only"})
    assert res.status_code == 200


def test_conflicting_symptoms(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Conflicting symptoms"})
    assert res.status_code == 200


def test_insufficient_evidence_case(monkeypatch, mock_agent_output):
    _setup(monkeypatch, mock_agent_output)
    res = client.post("/chat", json={"question": "Insufficient evidence"})
    assert res.status_code == 200
