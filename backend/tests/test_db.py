import json
import sqlite3

from grapevine import db
from grapevine.runs import words


def test_words_of_each_session_join_the_one_list(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT NOT NULL, recorded_at TEXT NOT NULL, vocabulary TEXT)"
    )
    old.execute(
        "INSERT INTO sessions (name, recorded_at, vocabulary) VALUES ('a', '2025-10-18', ?)", (json.dumps(["EZoo"]),)
    )
    old.execute(
        "INSERT INTO sessions (name, recorded_at, vocabulary) VALUES ('b', '2025-10-19', ?)",
        (json.dumps(["Glow", "ezoo"]),),
    )
    old.commit()
    old.close()

    conn = db.connect(path)

    assert words(conn) == ["EZoo", "Glow"]
    assert "vocabulary" not in {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}


def test_an_old_database_gets_each_migration_once(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT NOT NULL, recorded_at TEXT NOT NULL, "
        "roster TEXT NOT NULL DEFAULT '[]')"
    )
    old.execute(
        "CREATE TABLE segments (id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL, start_ms INTEGER NOT NULL, "
        "end_ms INTEGER NOT NULL, speaker_label TEXT NOT NULL, person_id INTEGER, speaker_confidence REAL, "
        "text TEXT NOT NULL, timings_json TEXT NOT NULL DEFAULT '{}')"
    )
    old.commit()
    old.close()

    db.connect(path).close()
    conn = db.connect(path)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    assert "speaker_manual" in db.columns(conn, "segments")
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1
    plan = conn.execute("EXPLAIN QUERY PLAN SELECT * FROM model_calls WHERE run_id = 1").fetchall()
    assert "model_calls_run_id" in plan[0]["detail"]
