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
from grapevine.llm import LLM, Call
from tests.fakes import FakeChat, empty_output, fact


@pytest.fixture
def client(conn, store, monkeypatch):
    monkeypatch.setattr(server, "conn", conn)
    monkeypatch.setattr(server, "active", {})
    monkeypatch.setattr(server, "stores", {})
    return TestClient(server.app)


def test_a_client_that_falls_behind_loses_its_socket(monkeypatch):
    monkeypatch.setattr(server, "clients", set())
    monkeypatch.setattr(server, "CLIENT_QUEUE_SIZE", 2)

    class SlowSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False
            self.sending = asyncio.Event()

        async def accept(self) -> None:
            pass

        async def send_text(self, text: str) -> None:
            self.sent.append(text)
            await self.sending.wait()

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> SlowSocket:
        socket = SlowSocket()
        handler = asyncio.create_task(server.websocket(socket))
        await asyncio.sleep(0)
        for seq in range(4):
            server.broadcast({"seq": seq})
            await asyncio.sleep(0)
        assert server.clients == set()
        socket.sending.set()
        await handler
        return socket

    socket = asyncio.run(scenario())
    assert socket.sent == ['{"seq": 0}'] and socket.closed


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


def test_unknown_things_are_not_found(client, store):
    run = f"/api/runs/{store.run_id}"
    assert client.post(f"{run}/segments/999/speaker", json={"new_name": "Felix Ward"}).status_code == 404
    assert client.patch(f"{run}/people/999", json={"name": "Felix Ward"}).status_code == 404
    assert client.post(f"/api/merge/{store.run_id}/prompts/999", json={"action": "keep"}).status_code == 404
    for path in ["changes", "model_calls"]:
        assert client.get(f"/api/runs/999/{path}").status_code == 404
        assert client.get(f"{run}/{path}").json() == []
    assert client.post("/api/runs/999/stop").status_code == 404
    assert client.post(f"{run}/stop").status_code == 409


def test_reading_a_run_does_not_replay_it(client, store):
    assert client.get(f"/api/runs/{store.run_id}").json()["session"]["name"] == "test"
    assert server.stores == {}
    assert client.get("/api/runs/999").status_code == 404


def test_bad_run_config_is_a_bad_request(client, store):
    body = {"session_id": 1, "source": "text"}
    assert client.post("/api/runs", json=body | {"config": {"bogus": 1}}).status_code == 400
    assert client.post("/api/runs", json=body | {"config": {"turn_rules": 5}}).status_code == 400


def test_a_replay_speed_must_be_above_zero(client):
    for speed in [0, -1]:
        assert client.post("/api/runs", json={"session_id": 999, "source": "file", "speed": speed}).status_code == 422


def test_a_run_cut_off_by_shutdown_fails_on_the_next_start(client, store, conn):
    run = server.ActiveRun(store.run_id, store, None)
    with client:
        server.set_status(conn, store.run_id, "running")
        server.active[store.run_id] = run
    status = conn.execute("SELECT status FROM runs WHERE id = ?", (store.run_id,)).fetchone()[0]
    assert run.stop.is_set() and status == "running"

    with client:
        pass
    assert conn.execute("SELECT status FROM runs WHERE id = ?", (store.run_id,)).fetchone()[0] == "failed"


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


def test_speaker_needs_one_known_person(client, store, monkeypatch):
    monkeypatch.setattr(server, "chat", lambda: FakeChat({}))
    monkeypatch.setattr(server, "llm_file_hash", lambda: "t")
    segment = store.add_segment(
        start_ms=0, end_ms=1000, speaker_label="s1", person_id=None, confidence=None, text="hi", timings={}
    )
    person = store.create_person(name="Mia Rao", aliases=[], gender="unknown")
    url = f"/api/runs/{store.run_id}/segments/{segment['id']}/speaker"
    for body in [{}, {"person_id": person["id"], "new_name": "Felix Ward"}, {"person_id": 999}]:
        assert client.post(url, json=body).status_code == 400
    assert client.post(url, json={"person_id": 999}).json() == {"detail": "no person 999"}
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


def test_a_run_that_is_going_reads_fixed_lines_in_its_pipeline(client, store):
    rereads = []

    class Pipeline:
        def reread(self, segment_ids, *, edited):
            rereads.append((segment_ids, edited))

    server.active[store.run_id] = server.ActiveRun(store.run_id, store, Pipeline())
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    segment = store.add_segment(
        start_ms=0, end_ms=900, speaker_label="spk0", person_id=None, confidence=None, text="Hi.", timings={}
    )

    client.patch(f"/api/runs/{store.run_id}/segments/{segment['id']}", json={"text": "Hello."})
    client.post(f"/api/runs/{store.run_id}/segments/{segment['id']}/speaker", json={"person_id": marcus})

    assert rereads == [([segment["id"]], True), ([segment["id"]], False)]


def test_the_llm_files_are_fingerprinted_off_the_event_loop(client, store, monkeypatch):
    fingerprinted_on_loop = []

    def llm_file_hash() -> str:
        try:
            asyncio.get_running_loop()
            fingerprinted_on_loop.append(True)
        except RuntimeError:
            fingerprinted_on_loop.append(False)
        return "t"

    monkeypatch.setattr(server, "chat", lambda: FakeChat({}))
    monkeypatch.setattr(server, "llm_file_hash", llm_file_hash)
    segment = store.add_segment(
        start_ms=0, end_ms=900, speaker_label="spk0", person_id=None, confidence=None, text="Hi.", timings={}
    )

    assert client.patch(f"/api/runs/{store.run_id}/segments/{segment['id']}", json={"text": "Hello."}).is_success
    assert client.post("/api/merge").is_success
    assert fingerprinted_on_loop == [False, False]


def test_two_edits_at_once_of_a_finished_run_both_save(conn, store, monkeypatch):
    class SlowChat(FakeChat):
        async def complete(self, request: dict, on_text=None) -> str:
            await asyncio.sleep(0.05)
            return await super().complete(request, on_text)

    rules = {
        "Seattle": {"facts": [fact("Marcus Hale works in Seattle.", ["speaker"])]},
        "Boston": {"facts": [fact("Marcus Hale works in Boston.", ["speaker"])]},
    }
    monkeypatch.setattr(server, "conn", conn)
    monkeypatch.setattr(server, "active", {})
    monkeypatch.setattr(server, "stores", {})
    monkeypatch.setattr(server, "rereading", asyncio.Lock())
    monkeypatch.setattr(server, "chat", lambda: SlowChat(rules))
    monkeypatch.setattr(server, "llm_file_hash", lambda: "t")
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    segment = store.add_segment(
        start_ms=0, end_ms=900, speaker_label="spk0", person_id=marcus, confidence=0.9, text="Hi.", timings={}
    )

    async def both() -> list:
        edits = [server.LineEdit(text="I work in Seattle."), server.LineEdit(text="I work in Boston.")]
        return await asyncio.gather(*(server.edit_segment(store.run_id, segment["id"], edit) for edit in edits))

    first, second = asyncio.run(both())

    assert first["text"] == second["text"] == "I work in Boston."
    fresh = server.open_store(conn, store.run_id)
    assert [f["text"] for f in fresh.graph.live_facts()] == ["Marcus Hale works in Boston."]


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


def test_a_new_roster_reaches_the_run_of_its_session(client, store, conn):
    class Running:
        pipeline = type("Pipeline", (), {"roster": []})()

    run = Running()
    server.active[store.run_id] = run
    other = conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('other', '2025-10-19')").lastrowid

    client.patch(f"/api/sessions/{other}", json={"roster": ["Felix Ward"]})
    assert run.pipeline.roster == []
    response = client.patch("/api/sessions/1", json={"roster": [" Mia  Rao ", "Dana Cole", ""]})

    assert response.json()["roster"] == ["Mia Rao", "Dana Cole"]
    assert run.pipeline.roster == ["Mia Rao", "Dana Cole"]


def test_new_words_reach_every_run_that_is_going(client, store):
    class Running:
        pipeline = type("Pipeline", (), {"vocabulary": []})()

    run = Running()
    server.active[store.run_id] = run

    response = client.put("/api/words", json={"words": [" EZoo ", "Glow", "ezoo"]})

    assert response.json() == ["EZoo", "Glow"]
    assert client.get("/api/words").json() == ["EZoo", "Glow"]
    assert run.pipeline.vocabulary == ["EZoo", "Glow"]


def test_the_debug_screen_gets_the_full_request_of_each_model_call(client, conn, store):
    chat = FakeChat({})
    llm = LLM(conn, chat, model_file_hash="t", reuse=False)
    for user in ("first", "second"):
        call = Call("merger", "m", "v1", "Merge them.", user, {"type": "object"}, 1)
        asyncio.run(llm.call(call, run_id=store.run_id, audio_ms=0))

    calls = client.get(f"/api/runs/{store.run_id}/model_calls").json()

    assert [c["request"] for c in calls] == chat.requests
    assert conn.execute("SELECT COUNT(*) FROM model_instructions").fetchone()[0] == 1
