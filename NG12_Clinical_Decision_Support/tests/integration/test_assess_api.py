from fastapi.testclient import TestClient

from app.main import app
from app.models import AgentOutput

client = TestClient(app)


def _set_rag(monkeypatch, results):
    from app import main

    class _Rag:
        def query(self, q, k):
            return results

    monkeypatch.setattr(main, "_rag_instance", _Rag())


def test_missing_patient_id():
    res = client.post("/assess", json={})
    assert res.status_code == 422


def test_invalid_patient_id():
    res = client.post("/assess", json={"patient_id": "does_not_exist"})
    assert res.status_code == 400


def test_empty_symptoms_list(monkeypatch):
    from app import tools

    monkeypatch.setattr(
        tools, "load_patients", lambda: {"p": {"patient_id": "p", "symptoms": []}}
    )
    res = client.post("/assess", json={"patient_id": "p"})
    assert res.status_code == 400


def test_known_high_risk_patient(monkeypatch, mock_urgent_output):
    from app import main

    _set_rag(monkeypatch, [])
    monkeypatch.setattr(main, "generate_assessment", lambda q, c: AgentOutput(**mock_urgent_output))

    res = client.post("/assess", json={"patient_id": "patient_001"})
    assert res.status_code == 200


def test_known_low_risk_patient(monkeypatch, mock_agent_output):
    from app import main

    _set_rag(monkeypatch, [])
    monkeypatch.setattr(main, "generate_assessment", lambda q, c: AgentOutput(**mock_agent_output))

    res = client.post("/assess", json={"patient_id": "patient_001"})
    assert res.status_code == 200


def test_young_patient_same_symptoms(monkeypatch, mock_agent_output):
    from app import main

    _set_rag(monkeypatch, [])
    monkeypatch.setattr(main, "generate_assessment", lambda q, c: AgentOutput(**mock_agent_output))

    res = client.post("/assess", json={"patient_id": "patient_002"})
    assert res.status_code == 200


def test_gemini_timeout_simulation(monkeypatch):
    from app import main

    def _raise_once():
        state = {"done": False}
        def _inner(question, chunks):
            if not state["done"]:
                state["done"] = True
                raise TimeoutError("Gemini timeout")
            return AgentOutput(
                assessment="Insufficient Evidence",
                reasoning="Insufficient evidence in retrieved NG12 text.",
                citations=[],
                confidence="low",
            )
        return _inner

    _set_rag(monkeypatch, [])
    monkeypatch.setattr(main, "generate_assessment", _raise_once())

    res = client.post("/assess", json={"patient_id": "patient_001"})
    assert res.status_code == 200


def test_vector_db_unavailable(monkeypatch, mock_agent_output):
    from app import main

    class _Rag:
        def query(self, q, k):
            raise RuntimeError("Vector DB unavailable")

    monkeypatch.setattr(main, "_rag_instance", _Rag())
    monkeypatch.setattr(main, "generate_assessment", lambda q, c: AgentOutput(**mock_agent_output))

    res = client.post("/assess", json={"patient_id": "patient_001"})
    assert res.status_code == 200
