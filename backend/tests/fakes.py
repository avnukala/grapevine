"""Fake models for tests. Each fake speaker talks at its own constant amplitude, so the fake speaker
detector and voice embedder can tell speakers apart from the audio alone."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from grapevine.audio import CHUNK_SAMPLES, SAMPLE_RATE
from grapevine.models import MAX_SPEAKERS, Models

SAMPLES_PER_FRAME = SAMPLE_RATE // 100


@dataclass(frozen=True)
class Line:
    speaker: int
    start_ms: int
    end_ms: int
    text: str


def amplitude(speaker: int) -> float:
    return 0.1 * (speaker + 1)


def render(lines: list[Line], total_ms: int) -> np.ndarray:
    audio = np.zeros(total_ms * SAMPLE_RATE // 1000, dtype=np.float32)
    for line in lines:
        audio[line.start_ms * SAMPLE_RATE // 1000 : line.end_ms * SAMPLE_RATE // 1000] = amplitude(line.speaker)
    return audio


async def chunks(audio: np.ndarray) -> AsyncIterator[np.ndarray]:
    for offset in range(0, len(audio), CHUNK_SAMPLES):
        yield audio[offset : offset + CHUNK_SAMPLES]


def speaker_of(level: float) -> int | None:
    return round(level * 10) - 1 if level > 0.05 else None


class FakeDetector:
    """Labels speakers by the order they first talk, like Nemotron. `merged` maps a speaker to another
    speaker whose label it gets, as when a real detector cannot tell two voices apart."""

    def __init__(self, merged: dict[int, int] | None = None) -> None:
        self.merged = merged or {}

    def start(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.order: dict[int, int] = {}

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self.pending = np.concatenate([self.pending, chunk])
        count = len(self.pending) // SAMPLES_PER_FRAME
        frames = np.zeros((count, MAX_SPEAKERS), dtype=np.float32)
        for i in range(count):
            speaker = speaker_of(float(self.pending[i * SAMPLES_PER_FRAME : (i + 1) * SAMPLES_PER_FRAME].mean()))
            if speaker is not None:
                speaker = self.merged.get(speaker, speaker)
                frames[i, self.order.setdefault(speaker, len(self.order))] = 0.9
        self.pending = self.pending[count * SAMPLES_PER_FRAME :]
        return frames

    def finish(self) -> np.ndarray:
        return np.zeros((0, MAX_SPEAKERS), dtype=np.float32)


class FakeEmbedder:
    def embed(self, audio: np.ndarray) -> np.ndarray:
        vector = np.zeros(MAX_SPEAKERS, dtype=np.float32)
        vector[speaker_of(float(np.median(audio)))] = 1.0
        return vector


class FakeTranscriber:
    """Returns the script line that best overlaps the audio it gets, found by the audio length and level."""

    def __init__(self, lines: list[Line]) -> None:
        self.lines = list(lines)

    def transcribe(self, audio: np.ndarray, names: list[str]) -> str:
        speaker = speaker_of(float(np.median(audio)))
        line = next(line for line in self.lines if line.speaker == speaker)
        self.lines.remove(line)
        return line.text


def empty_output() -> dict:
    return {"speaker_name": None, "people": [], "events": [], "facts": [], "relationships": []}


def fact(text: str, people: list[str], **fields: object) -> dict:
    return {
        "kind": "personal",
        "text": text,
        "people": people,
        "when_text": None,
        "start_date": None,
        "end_date": None,
        "event": None,
        "group": None,
        "revises": None,
        "disputes": None,
        **fields,
    }


class FakeChat:
    """Answers each extractor call by the first rule whose key appears in the new turn."""

    def __init__(
        self,
        rules: dict[str, dict],
        prompts: list[dict] | None = None,
        merged: dict | None = None,
        reviewed: dict | None = None,
    ) -> None:
        self.rules = rules
        self.prompts = prompts or []
        self.merged = merged or {"relationships": [], "conflicts": [], "same_person": []}
        self.reviewed = reviewed or {
            "room": {},
            "other": [],
            "known": {},
            "new": [],
            "happenings": [],
            "members": [],
        }
        self.requests: list[dict] = []

    async def complete(self, request: dict, on_text=None) -> str:
        self.requests.append(request)
        user = request["messages"][1]["content"]
        match request["response_format"]["json_schema"]["name"]:
            case "prompter":
                return json.dumps({"prompts": self.prompts})
            case "summarizer":
                return json.dumps({"stories": [], "relationships": []})
            case "merger":
                return json.dumps(self.merged)
            case "reviewer":
                return json.dumps(self.reviewed)
        new_turn = user.split("NEW TURN", 1)[1]
        for key, output in self.rules.items():
            if key in new_turn:
                return json.dumps({**empty_output(), **output})
        return json.dumps(empty_output())


def fake_models(lines: list[Line], chat: FakeChat, merged: dict[int, int] | None = None) -> Models:
    return Models(detector=FakeDetector(merged), embedder=FakeEmbedder(), transcriber=FakeTranscriber(lines), chat=chat)
