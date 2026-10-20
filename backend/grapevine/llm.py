"""Model calls with a record of each one. A call with the same input as a stored call reuses its response,
so an unchanged replay gives the same result."""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

from grapevine.models import ChatModel

Worker = Literal["extractor", "prompter", "summarizer", "merger", "judge"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Call:
    worker: Worker
    model: str
    prompt_version: str
    system: str
    user: str
    schema: dict
    seed: int
    max_tokens: int = 1500


@dataclass(frozen=True)
class Result:
    output: dict | None
    model_call_id: int


class LLM:
    def __init__(self, conn: sqlite3.Connection, chat: ChatModel, *, model_file_hash: str, reuse: bool) -> None:
        self.conn = conn
        self.chat = chat
        self.model_file_hash = model_file_hash
        self.reuse = reuse

    async def call(self, call: Call, *, run_id: int, audio_ms: int) -> Result:
        """Returns the parsed JSON output, or None when the model returned something that is not valid JSON."""
        request = {
            "model": call.model,
            "messages": [{"role": "system", "content": call.system}, {"role": "user", "content": call.user}],
            "temperature": 0,
            "seed": call.seed,
            "max_tokens": call.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": call.worker, "schema": call.schema, "strict": True},
            },
        }
        input_hash = hashlib.sha256(
            json.dumps([request, call.prompt_version, self.model_file_hash], sort_keys=True).encode()
        ).hexdigest()

        stored = None
        if self.reuse:
            stored = self.conn.execute(
                "SELECT response_json FROM model_calls WHERE input_hash = ? ORDER BY id LIMIT 1", (input_hash,)
            ).fetchone()
        started = time.monotonic()
        content = stored[0] if stored else await self.chat.complete(request)
        latency_ms = 0 if stored else int((time.monotonic() - started) * 1000)

        cursor = self.conn.execute(
            "INSERT INTO model_calls (run_id, worker, model, model_file_hash, prompt_version, seed, input_hash, "
            "request_json, response_json, latency_ms, cached, audio_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                call.worker,
                call.model,
                self.model_file_hash,
                call.prompt_version,
                call.seed,
                input_hash,
                json.dumps(request),
                content,
                latency_ms,
                int(stored is not None),
                audio_ms,
            ),
        )
        try:
            output = json.loads(content)
        except json.JSONDecodeError:
            log.warning("model call %s returned invalid JSON", cursor.lastrowid)
            output = None
        return Result(output, cursor.lastrowid)
