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
