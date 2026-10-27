import pytest
from fastapi.testclient import TestClient

from grapevine import server


@pytest.fixture
def client(conn, store, monkeypatch):
    monkeypatch.setattr(server, "conn", conn)
    monkeypatch.setattr(server, "active", {})
    return TestClient(server.app)


def test_unknown_session_is_not_found(client):
    assert client.patch("/api/sessions/999", json={"name": "Dana Cole"}).status_code == 404
    assert client.get("/api/sessions/999/audio").status_code == 404
    assert client.post("/api/runs", json={"session_id": 999, "source": "text"}).status_code == 404


def test_accept_needs_a_finished_run(client, store):
    assert client.post("/api/runs/999/accept").status_code == 404
    server.active[store.run_id] = object()
    assert client.post(f"/api/runs/{store.run_id}/accept").status_code == 409
    del server.active[store.run_id]
    assert client.post(f"/api/runs/{store.run_id}/accept").json()["accepted"] == 1
