import os
import tempfile
from pathlib import Path

import pytest

# Importing the server opens GRAPEVINE_DB, so the tests point it away from the real database first.
os.environ["GRAPEVINE_DB"] = str(Path(tempfile.mkdtemp()) / "server.db")

from grapevine import db
from grapevine.store import Store


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "test.db")
    yield connection
    connection.close()


@pytest.fixture
def store(conn):
    session_id = conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')").lastrowid
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (?, 'replay', '{}', 'test')",
        (session_id,),
    ).lastrowid
    return Store(conn, run_id)
