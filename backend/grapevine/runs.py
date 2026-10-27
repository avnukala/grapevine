"""Sessions and runs: create them, load the models a run needs, and run the pipeline."""

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path

import numpy as np
import openai

from grapevine.config import DATA_DIR, LLM_BASE_URL, LLM_MODEL_PATH, ROOT, RunConfig, code_version
from grapevine.graph import Record
from grapevine.llm import LLM
from grapevine.models import Models, NemotronDetector, OpenAIChat, WavLMEmbedder, WhisperTranscriber, model_file_hash
from grapevine.pipeline import Pipeline
from grapevine.store import Broadcast, Store

log = logging.getLogger(__name__)


def session(conn: sqlite3.Connection, session_id: int) -> Record:
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise KeyError(f"no session {session_id}")
    return session_record(row)


def session_record(row: sqlite3.Row) -> Record:
    return {**dict(row), "roster": json.loads(row["roster"])}


def create_session(
    conn: sqlite3.Connection,
    *,
    name: str,
    recorded_at: str,
    audio_uri: str | None,
    intro_end_ms: int | None,
    roster: list[str],
) -> Record:
    cursor = conn.execute(
        "INSERT INTO sessions (name, recorded_at, audio_uri, intro_end_ms, roster) VALUES (?, ?, ?, ?, ?)",
        (name, recorded_at, audio_uri, intro_end_ms, json.dumps(roster)),
    )
    return session(conn, cursor.lastrowid)


def audio_path(audio_uri: str) -> Path:
    path = Path(audio_uri).expanduser()
    return path if path.is_absolute() else ROOT / path


def session_for_audio(conn: sqlite3.Connection, path: Path) -> Record:
    """Finds the session of an audio file, or creates it. A sample with a `.labels.json` file next to it
    gets its name, date, end of introductions, and roster from the labels and script."""
    uri = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
    row = conn.execute("SELECT id FROM sessions WHERE audio_uri = ? ORDER BY id LIMIT 1", (uri,)).fetchone()
    if row:
        return session(conn, row[0])
    stem = path.name.rsplit(".", 1)[0]
    labels_path, script_path = path.with_name(f"{stem}.labels.json"), path.with_name(f"{stem}.script.json")
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    script = json.loads(script_path.read_text()) if script_path.exists() else {}
    return create_session(
        conn,
        name=script.get("name", stem),
        recorded_at=script.get("recorded_at", date.today().isoformat()),
        audio_uri=uri,
        intro_end_ms=labels.get("intro_end_ms"),
        roster=list(script.get("voices", {})),
    )


def create_run(conn: sqlite3.Connection, *, session_id: int | None, kind: str, config: RunConfig) -> int:
    cursor = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (?, ?, ?, ?)",
        (session_id, kind, json.dumps(config.to_json()), code_version()),
    )
    return cursor.lastrowid


def set_status(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))


def open_store(conn: sqlite3.Connection, run_id: int, broadcast: Broadcast | None = None) -> Store:
    """A store for a run that already has changes, with its graph rebuilt from them."""
    store = Store(conn, run_id, broadcast)
    for row in conn.execute("SELECT op_json, audio_ms FROM changes WHERE run_id = ? ORDER BY seq", (run_id,)):
        store.graph.apply(json.loads(row[0]))
        store.audio_ms = row[1]
    return store


@cache
def detector(model_id: str, mode: str, device: str) -> NemotronDetector:
    return NemotronDetector(model_id, mode=mode, device=device)


@cache
def embedder(model_id: str, device: str) -> WavLMEmbedder:
    return WavLMEmbedder(model_id, device=device)


@cache
def transcriber(model_id: str, language: str) -> WhisperTranscriber:
    return WhisperTranscriber(model_id, language=language)


@cache
def chat() -> OpenAIChat:
    return OpenAIChat(LLM_BASE_URL)


@cache
def llm_file_hash() -> str:
    """Fingerprints LLM_MODEL_PATH, or else the one LM Studio model folder whose name contains the model id."""
    if LLM_MODEL_PATH:
        return model_file_hash(LLM_MODEL_PATH)
    model = RunConfig().extract_model.lower()
    folders = [f for f in Path("~/.lmstudio/models").expanduser().glob("*/*") if model in f.name.lower()]
    return model_file_hash(str(folders[0])) if len(folders) == 1 else "unknown"


def load_models(config: RunConfig) -> Models:
    return Models(
        detector=detector(config.diarizer, config.diarizer_mode, config.diarizer_device),
        embedder=embedder(config.embedder, config.embedder_device),
        transcriber=transcriber(config.transcriber, config.language),
        chat=chat(),
    )


@dataclass
class ActiveRun:
    """A run whose pipeline is still going."""

    run_id: int
    store: Store
    pipeline: Pipeline
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None


async def run_pipeline(conn: sqlite3.Connection, active: ActiveRun, chunks: AsyncIterator[np.ndarray]) -> None:
    try:
        await active.pipeline.run(chunks)
    except Exception as error:
        log.exception("run %s failed", active.run_id)
        set_status(conn, active.run_id, "failed")
        active.store.notify({"type": "status", "status": "failed", "error": failure(error)})
        raise
    status = "stopped" if active.stop.is_set() else "done"
    set_status(conn, active.run_id, status)
    active.store.notify({"type": "status", "status": status})


def failure(error: Exception) -> str:
    if isinstance(error, openai.APIConnectionError):
        return f"The run stopped because no model answered at {LLM_BASE_URL}. Load the models in LM Studio, then start a new run."
    return f"The run stopped: {error}"


def start_pipeline(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    session: Record,
    config: RunConfig,
    models: Models,
    broadcast: Broadcast | None,
) -> ActiveRun:
    store = Store(conn, run_id, broadcast)
    llm = LLM(conn, models.chat, model_file_hash=llm_file_hash(), reuse=config.reuse_model_calls)
    pipeline = Pipeline(
        store,
        models,
        llm,
        config,
        today=session["recorded_at"][:10],
        intro_end_ms=session["intro_end_ms"],
        roster=session["roster"],
    )
    return ActiveRun(run_id, store, pipeline)


def recordings_dir() -> Path:
    return DATA_DIR / "audio"
