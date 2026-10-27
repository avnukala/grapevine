import asyncio
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from grapevine import server
from grapevine.extract import Held, Teller, apply_output
from tests.fakes import FakeChat, empty_output, fact


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


def test_bad_run_config_is_a_bad_request(client, store):
    body = {"session_id": 1, "source": "text"}
    assert client.post("/api/runs", json=body | {"config": {"bogus": 1}}).status_code == 400
    assert client.post("/api/runs", json=body | {"config": {"turn_rules": 5}}).status_code == 400


def test_merge_waits_for_the_run(client, store):
    server.active[store.run_id] = object()
    assert client.post("/api/merge").status_code == 409


def test_two_starts_at_once_start_one_run(conn, store, monkeypatch):
    monkeypatch.setattr(server, "conn", conn)
    monkeypatch.setattr(server, "active", {})

    def load_models(config):
        time.sleep(0.2)
        raise RuntimeError("no models in tests")

    monkeypatch.setattr(server, "load_models", load_models)
    body = server.NewRun(session_id=1, source="text")

    async def both() -> list:
        return await asyncio.gather(server.new_run(body), server.new_run(body), return_exceptions=True)

    first, second = asyncio.run(both())
    assert isinstance(first, RuntimeError)
    assert isinstance(second, HTTPException) and second.status_code == 409
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1


def test_blank_or_long_names_are_rejected(client, store):
    segment = store.add_segment(
        start_ms=0, end_ms=1000, speaker_label="s1", person_id=None, confidence=None, text="hi", timings={}
    )
    for name in ["   ", "a" * 101]:
        answer = client.post(f"/api/runs/{store.run_id}/segments/{segment['id']}/speaker", json={"new_name": name})
        assert answer.status_code == 422
        assert client.post("/api/sessions", json={"name": name}).status_code == 422
    assert store.conn.execute("SELECT count(*) FROM people").fetchone()[0] == 0


def test_speaker_needs_one_known_person(client, store):
    segment = store.add_segment(
        start_ms=0, end_ms=1000, speaker_label="s1", person_id=None, confidence=None, text="hi", timings={}
    )
    person = store.create_person(name="Mia Rao", aliases=[], gender="unknown")
    url = f"/api/runs/{store.run_id}/segments/{segment['id']}/speaker"
    for body in [{}, {"person_id": person["id"], "new_name": "Felix Ward"}, {"person_id": 999}]:
        assert client.post(url, json=body).status_code == 400
    assert (
        client.post(f"/api/runs/{store.run_id}/segments/999/speaker", json={"new_name": "Felix Ward"}).status_code
        == 400
    )
    assert store.conn.execute("SELECT count(*) FROM people").fetchone()[0] == 1
    assert client.post(url, json={"new_name": "Felix Ward"}).json()["person_id"] != person["id"]


def test_upload_rejects_a_file_that_is_not_audio(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "recordings_dir", lambda: tmp_path / "audio")
    answer = client.post("/api/sessions/upload", files={"file": ("talk.mp3", b"not audio")})
    assert answer.status_code == 400
    assert list((tmp_path / "audio").iterdir()) == []


def test_upload_outside_the_checkout_keeps_an_absolute_path(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "recordings_dir", lambda: tmp_path)
    wav = tmp_path / "source.wav"
    sf.write(wav, np.zeros(1600, dtype=np.float32), 16000)
    answer = client.post("/api/sessions/upload", files={"file": ("talk.wav", wav.read_bytes())})
    assert answer.status_code == 200
    assert Path(answer.json()["audio_uri"]).is_absolute()


def test_an_edited_line_of_a_finished_run_gets_new_facts(client, store, monkeypatch):
    rules = {"Seattle": {"facts": [fact("Marcus Hale works in Seattle.", ["speaker"])]}}
    monkeypatch.setattr(server, "chat", lambda: FakeChat(rules))
    monkeypatch.setattr(server, "llm_file_hash", lambda: "t")
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    segment = store.add_segment(
        start_ms=0,
        end_ms=900,
        speaker_label="spk0",
        person_id=marcus,
        confidence=0.9,
        text="I work in Denver.",
        timings={},
    )
    told = {**empty_output(), "facts": [fact("Marcus Hale works in Denver.", ["speaker"])]}
    apply_output(store, Held(told, Teller(segment["id"], marcus, 0.9, None)))

    response = client.patch(f"/api/runs/{store.run_id}/segments/{segment['id']}", json={"text": "I work in Seattle."})

    assert response.json()["text"] == "I work in Seattle."
    fresh = server.store_for(store.run_id)
    assert [f["text"] for f in fresh.graph.live_facts()] == ["Marcus Hale works in Seattle."]


def test_a_line_moved_to_another_speaker_after_a_run_gets_new_facts(client, store, monkeypatch):
    rules = {"Denver": {"facts": [fact("Felix Ward works in Denver.", ["speaker"])]}}
    monkeypatch.setattr(server, "chat", lambda: FakeChat(rules))
    monkeypatch.setattr(server, "llm_file_hash", lambda: "t")
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    felix = store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")["id"]
    segment = store.add_segment(
        start_ms=0,
        end_ms=900,
        speaker_label="spk0",
        person_id=marcus,
        confidence=0.9,
        text="I work in Denver.",
        timings={},
    )
    told = {**empty_output(), "facts": [fact("Marcus Hale works in Denver.", ["speaker"])]}
    apply_output(store, Held(told, Teller(segment["id"], marcus, 0.9, None)))

    body = {"person_id": felix, "every_line": False}
    client.post(f"/api/runs/{store.run_id}/segments/{segment['id']}/speaker", json=body)

    [new] = server.store_for(store.run_id).graph.live_facts()
    assert (new["text"], new["told_by_person_id"]) == ("Felix Ward works in Denver.", felix)


def test_renaming_a_roster_person_renames_them_on_the_roster(client, store, conn):
    conn.execute("UPDATE sessions SET roster = ? WHERE id = 1", ('["Miaely Rao", "Dana Cole"]',))
    mia = store.create_person(name="Miaely Rao", aliases=["Miaely"], gender="female")["id"]
    store.show_person(mia)

    response = client.patch(f"/api/runs/{store.run_id}/people/{mia}", json={"name": "Mia Rao"})

    assert response.json()["name"] == "Mia Rao"
    assert server.session(conn, 1)["roster"] == ["Mia Rao", "Dana Cole"]
    assert client.patch(f"/api/runs/{store.run_id}/people/{mia}", json={"name": " "}).status_code == 422


def test_audio_files_outside_the_checkout_are_listed(client, tmp_path, monkeypatch):
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "dinner.mp3").write_bytes(b"")
    monkeypatch.setattr(server, "DATA_DIR", tmp_path)
    monkeypatch.setattr(server, "recordings_dir", lambda: tmp_path / "audio")

    assert client.get("/api/audio-files").json() == [str((tmp_path / "samples" / "dinner.mp3").resolve())]
