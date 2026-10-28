"""SQLite schema and connection. Every write goes through `Store` in `store.py`."""

import hashlib
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    audio_uri TEXT,
    notes TEXT,
    intro_end_ms INTEGER,
    roster TEXT NOT NULL DEFAULT '[]'
);
-- Words the transcript gets wrong, such as nicknames, places, and slang. One list for every session.
CREATE TABLE IF NOT EXISTS words (
    word TEXT PRIMARY KEY COLLATE NOCASE
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    kind TEXT NOT NULL CHECK (kind IN ('live', 'replay', 'merge')),
    config_json TEXT NOT NULL,
    code_version TEXT NOT NULL,
    accepted INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
-- The system prompt and response format of model requests. Many calls share the same few, and each is kilobytes.
CREATE TABLE IF NOT EXISTS model_instructions (
    hash TEXT PRIMARY KEY,
    system TEXT NOT NULL,
    response_format_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    worker TEXT NOT NULL,
    model TEXT NOT NULL,
    model_file_hash TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    -- The request without its system prompt and response format, which model_instructions holds.
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    cached INTEGER NOT NULL DEFAULT 0,
    audio_ms INTEGER,
    instructions_hash TEXT REFERENCES model_instructions(hash)
);
CREATE INDEX IF NOT EXISTS model_calls_input_hash ON model_calls(input_hash);
CREATE INDEX IF NOT EXISTS model_calls_run_id ON model_calls(run_id);
CREATE TABLE IF NOT EXISTS changes (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    seq INTEGER NOT NULL,
    audio_ms INTEGER NOT NULL,
    segment_id INTEGER,
    model_call_id INTEGER,
    op_json TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    speaker_label TEXT NOT NULL,
    person_id INTEGER REFERENCES people(id),
    speaker_confidence REAL,
    speaker_manual INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    timings_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS speakers (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    speaker_label TEXT NOT NULL,
    person_id INTEGER REFERENCES people(id),
    voice_embedding BLOB,
    PRIMARY KEY (run_id, speaker_label)
);
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    gender TEXT NOT NULL DEFAULT 'unknown' CHECK (gender IN ('male', 'female', 'unknown')),
    merged_into_id INTEGER REFERENCES people(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL CHECK (kind IN ('personal', 'relationship', 'story', 'membership', 'correction')),
    text TEXT NOT NULL,
    detail TEXT,
    told_by_person_id INTEGER REFERENCES people(id),
    when_text TEXT,
    start_date TEXT,
    end_date TEXT,
    date_precision TEXT CHECK (date_precision IN ('year', 'season', 'month', 'day')),
    event_id INTEGER REFERENCES events(id),
    confidence REAL NOT NULL DEFAULT 1,
    disputed INTEGER NOT NULL DEFAULT 0,
    revised_by_fact_id INTEGER REFERENCES facts(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS fact_people (
    fact_id INTEGER NOT NULL REFERENCES facts(id),
    person_id INTEGER NOT NULL REFERENCES people(id),
    role TEXT NOT NULL,
    PRIMARY KEY (fact_id, person_id)
);
CREATE TABLE IF NOT EXISTS fact_segments (
    fact_id INTEGER NOT NULL REFERENCES facts(id),
    segment_id INTEGER NOT NULL REFERENCES segments(id),
    PRIMARY KEY (fact_id, segment_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    name TEXT NOT NULL,
    when_text TEXT,
    start_date TEXT,
    end_date TEXT,
    date_precision TEXT CHECK (date_precision IN ('year', 'season', 'month', 'day')),
    place TEXT
);
CREATE TABLE IF NOT EXISTS event_people (
    event_id INTEGER NOT NULL REFERENCES events(id),
    person_id INTEGER NOT NULL REFERENCES people(id),
    role TEXT NOT NULL,
    PRIMARY KEY (event_id, person_id)
);
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    description TEXT
);
CREATE TABLE IF NOT EXISTS group_members (
    group_id INTEGER NOT NULL REFERENCES groups(id),
    person_id INTEGER NOT NULL REFERENCES people(id),
    since TEXT,
    until TEXT,
    source_fact_id INTEGER REFERENCES facts(id),
    PRIMARY KEY (group_id, person_id, source_fact_id)
);
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    person_a_id INTEGER NOT NULL REFERENCES people(id),
    person_b_id INTEGER NOT NULL REFERENCES people(id),
    category TEXT NOT NULL CHECK (category IN ('romantic', 'family', 'friend', 'professional', 'other')),
    label TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL CHECK (status IN ('current', 'past')),
    met_on TEXT,
    met_event_id INTEGER REFERENCES events(id),
    closeness INTEGER,
    recency INTEGER,
    sentiment INTEGER,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (run_id, person_a_id, person_b_id, category)
);
CREATE TABLE IF NOT EXISTS relationship_evidence (
    relationship_id INTEGER NOT NULL REFERENCES relationships(id),
    fact_id INTEGER NOT NULL REFERENCES facts(id),
    PRIMARY KEY (relationship_id, fact_id)
);
CREATE TABLE IF NOT EXISTS prompts (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    text TEXT NOT NULL,
    reason TEXT NOT NULL,
    person_ids TEXT NOT NULL DEFAULT '[]',
    shown_at_ms INTEGER NOT NULL,
    dismissed_at_ms INTEGER,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
"""


def save_request(conn: sqlite3.Connection, request: dict) -> tuple[str, str]:
    """Saves the system prompt and response format of a model request once. Returns their hash and the rest of
    the request as JSON."""
    system, *messages = request["messages"]
    response_format = json.dumps(request["response_format"])
    instructions_hash = hashlib.sha256(json.dumps([system["content"], response_format]).encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO model_instructions (hash, system, response_format_json) VALUES (?, ?, ?)",
        (instructions_hash, system["content"], response_format),
    )
    rest = {key: value for key, value in request.items() if key != "response_format"}
    return instructions_hash, json.dumps({**rest, "messages": messages})


def full_request(row: sqlite3.Row) -> dict:
    """Rebuilds a model request that `save_request` split, from a model_calls row joined to model_instructions."""
    request = json.loads(row["request_json"])
    request["messages"].insert(0, {"role": "system", "content": row["system"]})
    return {**request, "response_format": json.loads(row["response_format_json"])}


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def add_session_roster(conn: sqlite3.Connection) -> None:
    if "roster" not in columns(conn, "sessions"):
        conn.execute("ALTER TABLE sessions ADD COLUMN roster TEXT NOT NULL DEFAULT '[]'")


def move_vocabulary_to_words(conn: sqlite3.Connection) -> None:
    """The word list was a list per session for a short time. Its words join the one list for all sessions."""
    if "vocabulary" not in columns(conn, "sessions"):
        return
    for (vocabulary,) in conn.execute("SELECT vocabulary FROM sessions").fetchall():
        conn.executemany("INSERT OR IGNORE INTO words (word) VALUES (?)", [(w,) for w in json.loads(vocabulary)])
    conn.execute("ALTER TABLE sessions DROP COLUMN vocabulary")


def add_segment_speaker_manual(conn: sqlite3.Connection) -> None:
    if "speaker_manual" not in columns(conn, "segments"):
        conn.execute("ALTER TABLE segments ADD COLUMN speaker_manual INTEGER NOT NULL DEFAULT 0")


def split_model_requests(conn: sqlite3.Connection) -> None:
    if "instructions_hash" not in columns(conn, "model_calls"):
        conn.execute("ALTER TABLE model_calls ADD COLUMN instructions_hash TEXT REFERENCES model_instructions(hash)")
    rows = conn.execute("SELECT id, request_json FROM model_calls WHERE instructions_hash IS NULL").fetchall()
    for model_call_id, request_json in rows:
        instructions_hash, rest = save_request(conn, json.loads(request_json))
        conn.execute(
            "UPDATE model_calls SET instructions_hash = ?, request_json = ? WHERE id = ?",
            (instructions_hash, rest, model_call_id),
        )


# Append new steps only. A database at user_version n has run the first n steps. Databases from before
# user_version start at 0 with some steps done, so each step checks first.
MIGRATIONS = (add_session_roster, move_vocabulary_to_words, add_segment_speaker_manual, split_model_requests)


def migrate(conn: sqlite3.Connection) -> None:
    """Runs the steps of MIGRATIONS that the database has not run yet, each in its own transaction."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for number, step in enumerate(MIGRATIONS[version:], start=version + 1):
        conn.execute("BEGIN IMMEDIATE")
        step(conn)
        conn.execute(f"PRAGMA user_version = {number}")
        conn.execute("COMMIT")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn
