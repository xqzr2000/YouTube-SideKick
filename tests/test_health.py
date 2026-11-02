from fastapi.testclient import TestClient

from sidekick.api.app import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_cors_allows_extension_origin():
    origin = "chrome-extension://abcdefghijklmnop"
    r = client.get("/health", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == origin


def test_cors_rejects_other_origin():
    r = client.get("/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in r.headers
