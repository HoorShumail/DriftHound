from fastapi.testclient import TestClient

from api import server


client = TestClient(server.app)


def test_status_returns_200():
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_drift_report_returns_404_when_no_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DRIFT_REPORTS_DIR", tmp_path)

    response = client.get("/api/drift-report")

    assert response.status_code == 404


def test_diagnose_returns_503_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(server, "DRIFT_REPORTS_DIR", "missing-reports")
    monkeypatch.setattr("agent.client.load_dotenv", lambda: None)

    response = client.post(
        "/api/diagnose",
        json={"drift_report": {"summary": {"drifted_features": []}}},
    )

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
