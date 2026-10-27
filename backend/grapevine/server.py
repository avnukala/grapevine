"""The HTTP API and the WebSocket that pushes each change to the client."""

import asyncio
import json
import logging
import shutil
from collections.abc import AsyncIterator
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import openai
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from grapevine import db
from grapevine.actions import ActionError, assign_speaker, pick_person, pick_speaker, settle_conflict
from grapevine.audio import file_chunks, mic_chunks
from grapevine.config import DATA_DIR, DB_PATH, LLM_BASE_URL, ROOT, RunConfig, run_config
from grapevine.graph import Record
from grapevine.llm import LLM
from grapevine.merge import merge, resolve_merge
from grapevine.runs import (
    ActiveRun,
    audio_path,
    chat,
    create_run,
    create_session,
    llm_file_hash,
    load_models,
    open_store,
    recordings_dir,
    run_pipeline,
    session,
    session_for_audio,
    session_record,
    set_status,
    start_pipeline,
)
from grapevine.store import Store

log = logging.getLogger(__name__)
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aiff"}

app = FastAPI(title="Grapevine")
# One connection for the whole server. Every route that uses it, or a running pipeline, is async, so it runs on
# the event loop with the pipelines and never on a worker thread at the same time.
conn = db.connect(DB_PATH)
# A run that was going when the last server stopped can never finish, so it would stay "running" forever.
conn.execute("UPDATE runs SET status = 'failed' WHERE status = 'running'")
clients: set[asyncio.Queue] = set()
active: dict[int, ActiveRun] = {}


def broadcast(message: Record) -> None:
    for queue in clients:
        queue.put_nowait(message)


def store_for(run_id: int) -> Store:
    if run_id in active:
        return active[run_id].store
    if conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
        raise HTTPException(404, f"no run {run_id}")
    return open_store(conn, run_id, broadcast)


def session_or_404(session_id: int) -> Record:
    try:
        return session(conn, session_id)
    except KeyError:
        raise HTTPException(404, f"no session {session_id}") from None


def threshold(run_id: int) -> float:
    config = json.loads(conn.execute("SELECT config_json FROM runs WHERE id = ?", (run_id,)).fetchone()[0])
    return config.get("voice_threshold", RunConfig().voice_threshold)


@app.websocket("/ws")
async def websocket(socket: WebSocket) -> None:
    await socket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    clients.add(queue)
    try:
        while True:
            await socket.send_text(json.dumps(await queue.get()))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(queue)


@app.get("/api/audio-files")
def audio_files() -> list[str]:
    files = [*(DATA_DIR / "samples").glob("*"), *recordings_dir().glob("*")]
    return sorted(str(f.relative_to(ROOT)) for f in files if f.suffix in AUDIO_SUFFIXES)


@app.get("/api/sessions")
async def sessions() -> list[Record]:
    rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
    runs = conn.execute("SELECT id, session_id, kind, accepted, status, created_at, code_version FROM runs").fetchall()
    return [{**session_record(s), "runs": [dict(r) for r in runs if r["session_id"] == s["id"]][::-1]} for s in rows]


class NewSession(BaseModel):
    name: str
    audio_uri: str | None = None
    recorded_at: str | None = None
    intro_end_ms: int | None = None
    roster: list[str] = []


@app.post("/api/sessions")
async def new_session(body: NewSession) -> Record:
    """A session for a recording that already has one returns that session."""
    if body.audio_uri:
        if not audio_path(body.audio_uri).is_file():
            raise HTTPException(400, f"no audio file at {body.audio_uri}")
        return session_for_audio(conn, audio_path(body.audio_uri))
    return create_session(
        conn,
        name=body.name,
        recorded_at=body.recorded_at or date.today().isoformat(),
        audio_uri=body.audio_uri,
        intro_end_ms=body.intro_end_ms,
        roster=clean_roster(body.roster),
    )


@app.post("/api/sessions/upload")
async def upload_session(file: UploadFile) -> Record:
    name = Path(file.filename or "recording").name
    if Path(name).suffix.lower() not in AUDIO_SUFFIXES:
        raise HTTPException(400, f"unsupported audio type {Path(name).suffix}")
    target = recordings_dir() / f"{datetime.now():%Y%m%d-%H%M%S}-{name}"
    target.parent.mkdir(parents=True, exist_ok=True)

    def copy() -> None:
        with target.open("wb") as out:
            shutil.copyfileobj(file.file, out)

    await asyncio.to_thread(copy)
    return create_session(
        conn,
        name=Path(name).stem,
        recorded_at=date.today().isoformat(),
        audio_uri=str(target.relative_to(ROOT)),
        intro_end_ms=None,
        roster=[],
    )


def clean_roster(names: list[str]) -> list[str]:
    """Typed names with spaces tidied, blanks dropped, and each name once."""
    return list(dict.fromkeys(n for name in names if (n := " ".join(name.split()))))


class SessionUpdate(BaseModel):
    name: str | None = None
    notes: str | None = None
    intro_end_ms: int | None = None
    roster: list[str] | None = None


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: int, body: SessionUpdate) -> Record:
    session_or_404(session_id)
    for key, value in body.model_dump(exclude_none=True).items():
        if key == "roster":
            value = json.dumps(clean_roster(value))
        conn.execute(f"UPDATE sessions SET {key} = ? WHERE id = ?", (value, session_id))
    return session(conn, session_id)


@app.get("/api/sessions/{session_id}/audio")
async def session_audio(session_id: int) -> FileResponse:
    uri = session_or_404(session_id)["audio_uri"]
    if not uri or not audio_path(uri).is_file():
        raise HTTPException(404, "this session has no audio")
    return FileResponse(audio_path(uri))


class NewRun(BaseModel):
    session_id: int
    source: Literal["file", "mic", "text"]
    speed: float | None = None
    config: dict = {}


async def idle(stop: asyncio.Event) -> AsyncIterator[np.ndarray]:
    await stop.wait()
    return
    yield


@app.post("/api/runs")
async def new_run(body: NewRun) -> Record:
    if active:
        raise HTTPException(409, f"run {next(iter(active))} is still going")
    current = session_or_404(body.session_id)
    if body.source == "file" and not current["audio_uri"]:
        raise HTTPException(400, "this session has no audio to replay")
    config = run_config(dict(body.config))
    run_id = create_run(
        conn, session_id=current["id"], kind="replay" if body.source == "file" else "live", config=config
    )
    models = await asyncio.to_thread(load_models, config)
    run = start_pipeline(conn, run_id=run_id, session=current, config=config, models=models, broadcast=broadcast)
    match body.source:
        case "file":
            chunks = file_chunks(audio_path(current["audio_uri"]), speed=body.speed, stop=run.stop)
        case "mic":
            recording = recordings_dir() / f"session-{current['id']}-run-{run_id}.wav"
            conn.execute(
                "UPDATE sessions SET audio_uri = ? WHERE id = ?", (str(recording.relative_to(ROOT)), current["id"])
            )
            chunks = mic_chunks(recording, stop=run.stop)
        case "text":
            run.pipeline.intro_end_ms = 0
            chunks = idle(run.stop)
    run.pipeline.partials = body.source == "mic" or body.speed is not None
    active[run_id] = run

    async def go() -> None:
        try:
            await run_pipeline(conn, run, chunks)
        finally:
            active.pop(run_id, None)

    run.task = asyncio.create_task(go())
    return run_record(run_id)


@app.get("/api/runs/active")
async def going_run() -> Record | None:
    """The run that is still going, if any. The server runs one at a time."""
    return run_record(next(iter(active))) if active else None


def run_record(run_id: int) -> Record:
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return {**dict(row), "config": json.loads(row["config_json"]), "active": run_id in active}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int) -> Record:
    store_for(run_id)
    record = run_record(run_id)
    record["session"] = session(conn, record["session_id"]) if record["session_id"] else None
    return record


@app.get("/api/runs/{run_id}/changes")
async def changes(run_id: int) -> list[Record]:
    rows = conn.execute(
        "SELECT seq, audio_ms, segment_id, model_call_id, op_json FROM changes WHERE run_id = ? ORDER BY seq", (run_id,)
    )
    return [{**dict(r), "op": json.loads(r["op_json"])} | {"op_json": None} for r in rows]


@app.get("/api/runs/{run_id}/model_calls")
async def model_calls(run_id: int) -> list[Record]:
    rows = conn.execute("SELECT * FROM model_calls WHERE run_id = ? ORDER BY id", (run_id,))
    return [
        {
            **{
                k: r[k]
                for k in (
                    "id",
                    "worker",
                    "model",
                    "model_file_hash",
                    "prompt_version",
                    "seed",
                    "input_hash",
                    "latency_ms",
                    "audio_ms",
                )
            },
            "cached": bool(r["cached"]),
            "request": json.loads(r["request_json"]),
            "response": r["response_json"],
        }
        for r in rows
    ]


def active_run(run_id: int) -> ActiveRun:
    if run_id not in active:
        raise HTTPException(409, f"run {run_id} is not going")
    return active[run_id]


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: int) -> Record:
    active_run(run_id).stop.set()
    return run_record(run_id)


@app.post("/api/runs/{run_id}/intro-done")
async def intro_done(run_id: int) -> Record:
    run = active_run(run_id)
    run.pipeline.end_intro()
    session_id = conn.execute("SELECT session_id FROM runs WHERE id = ?", (run_id,)).fetchone()[0]
    conn.execute("UPDATE sessions SET intro_end_ms = ? WHERE id = ?", (run.pipeline.intro_end_ms, session_id))
    return run_record(run_id)


@app.post("/api/runs/{run_id}/accept")
async def accept_run(run_id: int) -> Record:
    row = conn.execute("SELECT session_id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no run {run_id}")
    if run_id in active:
        raise HTTPException(409, f"run {run_id} is still going")
    session_id = row[0]
    conn.execute("BEGIN")
    conn.execute("UPDATE runs SET accepted = 0 WHERE session_id = ?", (session_id,))
    conn.execute("UPDATE runs SET accepted = 1 WHERE id = ?", (run_id,))
    conn.execute("COMMIT")
    return run_record(run_id)


class TextTurn(BaseModel):
    speaker: str
    text: str


@app.post("/api/runs/{run_id}/text")
async def text_turn(run_id: int, body: TextTurn) -> Record:
    if not body.text.strip() or not body.speaker.strip():
        raise HTTPException(400, "a text turn needs a speaker and text")
    return await active_run(run_id).pipeline.add_text(body.speaker.strip(), body.text.strip())


class Answer(BaseModel):
    person_id: int | None = None
    new_name: str | None = None
    nickname: str | None = None
    keep_fact_id: int | None = None


@app.post("/api/runs/{run_id}/prompts/{prompt_id}/answer")
async def answer_prompt(run_id: int, prompt_id: int, body: Answer) -> Record:
    store = store_for(run_id)
    prompt = store.graph.prompts.get(prompt_id)
    if prompt is None:
        raise HTTPException(404, f"no prompt {prompt_id} in run {run_id}")
    try:
        match prompt["reason"]:
            case "which_person":
                pick_person(store, prompt_id, person_id=body.person_id, new_name=body.new_name)
            case "which_speaker":
                pick_speaker(
                    store, prompt_id, person_id=body.person_id, nickname=body.nickname, threshold=threshold(run_id)
                )
            case "conflict":
                settle_conflict(store, prompt_id, keep_fact_id=body.keep_fact_id)
            case _:
                store.dismiss_prompt(prompt_id)
    except ActionError as error:
        raise HTTPException(400, str(error)) from error
    return store.graph.prompts[prompt_id]


@app.post("/api/runs/{run_id}/prompts/{prompt_id}/dismiss")
async def dismiss_prompt(run_id: int, prompt_id: int) -> Record:
    store = store_for(run_id)
    if prompt_id not in store.graph.prompts:
        raise HTTPException(404, f"no prompt {prompt_id} in run {run_id}")
    store.dismiss_prompt(prompt_id)
    return store.graph.prompts[prompt_id]


class SpeakerAnswer(BaseModel):
    person_id: int | None = None
    new_name: str | None = None
    every_line: bool = False


@app.post("/api/runs/{run_id}/segments/{segment_id}/speaker")
async def set_segment_speaker(run_id: int, segment_id: int, body: SpeakerAnswer) -> Record:
    store = store_for(run_id)
    person_id = body.person_id
    if person_id is None:
        if not body.new_name:
            raise HTTPException(400, "pick a person or give a new name")
        person_id = store.create_person(name=body.new_name, aliases=[], gender="unknown")["id"]
    try:
        changed = assign_speaker(
            store, segment_id, person_id=person_id, every_line=body.every_line, threshold=threshold(run_id)
        )
    except ActionError as error:
        raise HTTPException(400, str(error)) from error
    if run_id in active:
        active[run_id].pipeline.reread(changed)
    return store.graph.segments[segment_id]


@app.delete("/api/runs/{run_id}/people/{person_id}")
async def remove_person(run_id: int, person_id: int) -> None:
    store = store_for(run_id)
    if person_id not in store.graph.people:
        raise HTTPException(404, f"person {person_id} is not in run {run_id}")
    store.remove_person(person_id)


@app.get("/api/people")
async def people() -> list[Record]:
    rows = conn.execute("SELECT * FROM people WHERE merged_into_id IS NULL ORDER BY name")
    return [{**dict(r), "aliases": json.loads(r["aliases"])} for r in rows]


@app.post("/api/merge")
async def new_merge() -> Record:
    config = run_config({})
    store = Store(conn, create_run(conn, session_id=None, kind="merge", config=config), broadcast)
    llm = LLM(conn, chat(), model_file_hash=llm_file_hash(), reuse=config.reuse_model_calls)
    try:
        await merge(store, llm, model=config.summary_model, seed=config.seed)
    except Exception as error:
        set_status(conn, store.run_id, "failed")
        if isinstance(error, openai.APIConnectionError):
            message = f"No model answered at {LLM_BASE_URL}. Load the models in LM Studio, then merge again."
            raise HTTPException(503, message) from error
        raise
    set_status(conn, store.run_id, "done")
    return run_record(store.run_id)


@app.get("/api/global")
async def global_graph() -> Record:
    row = conn.execute(
        "SELECT id FROM runs WHERE kind = 'merge' AND status = 'done' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return {"run": None}
    return {"run": run_record(row[0])}


class MergeAnswer(BaseModel):
    action: Literal["keep", "keep_both", "remove_edge", "merge_people", "keep_apart"]
    fact_id: int | None = None


@app.post("/api/merge/{run_id}/prompts/{prompt_id}")
async def answer_merge(run_id: int, prompt_id: int, body: MergeAnswer) -> Record:
    store = store_for(run_id)
    try:
        resolve_merge(store, prompt_id, action=body.action, fact_id=body.fact_id)
    except ActionError as error:
        raise HTTPException(400, str(error)) from error
    return store.graph.prompts[prompt_id]


@app.on_event("shutdown")
async def stop_all() -> None:
    for run in active.values():
        run.stop.set()
        set_status(conn, run.run_id, "stopped")
