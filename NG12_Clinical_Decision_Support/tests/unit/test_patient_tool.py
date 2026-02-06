from app.tools import get_patient


def test_valid_patient_id():
    patient = get_patient("patient_001")
    assert patient


def test_invalid_patient_id():
    patient = get_patient("missing")
    assert patient == {}


def test_missing_fields():
    patient = get_patient("patient_001")
    assert "symptoms" in patient


def test_empty_symptoms_list(monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "load_patients", lambda: {"p": {"patient_id": "p", "symptoms": []}})
    patient = get_patient("p")
    assert patient.get("symptoms") == []


def test_patient_with_missing_id(monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "load_patients", lambda: {None: {"symptoms": ["x"]}})
    patient = get_patient("missing")
    assert patient == {}


def test_patient_with_non_list_symptoms(monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "load_patients", lambda: {"p": {"patient_id": "p", "symptoms": "not-list"}})
    patient = get_patient("p")
    assert patient.get("symptoms") == "not-list"
