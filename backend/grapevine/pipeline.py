"""One pass over the audio of a session. A live microphone and a replayed file use this same path.

Audio flows through three stages that each keep turn order: speaker detection cuts turns, the
transcriber writes segments, and the extractor turns segments into facts. The prompter runs beside the
extractor. Each stage reads the graph only as the stage before it left it, so a replay is deterministic.
"""

import asyncio
import logging
import re
import time
from bisect import bisect_left
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace

import numpy as np

from grapevine.audio import SAMPLE_RATE
from grapevine.config import RunConfig
from grapevine.extract import Held, Teller, apply_output, extraction_call, held_payload
from grapevine.llm import LLM
from grapevine.models import Models
from grapevine.prompter import prompter_call, show_prompt, suggest
from grapevine.store import Store, name_matches
from grapevine.summarize import summarize
from grapevine.turns import Turn, TurnBuilder

log = logging.getLogger(__name__)

ACTIVITY_SILENCE_MS = 300
SELF_INTRO = re.compile(r"\b(?:I'm|I am|my name is|name's|call me)\s+([A-Z][a-z]+)")


def said_name(text: str, model_name: str | None) -> str | None:
    """The name speakers give for themselves. The model's name counts only when the turn says it."""
    first_names = [m[1] for m in SELF_INTRO.finditer(text)]
    if not first_names:
        return None
    if model_name and model_name.split()[0] in first_names and all(w in text for w in model_name.split()):
        return model_name
    first = first_names[0]
    return max(re.findall(rf"\b{first}(?:\s+[A-Z][a-z]+)?", text), key=len)


@dataclass(frozen=True)
class SegmentJob:
    segment_id: int
    heard_at: float


@dataclass(frozen=True)
class PromptJob:
    trigger: str
    at_ms: int


class AudioBuffer:
    def __init__(self) -> None:
        self.samples = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)
        self.size = 0
        self.arrival_ms: list[int] = []
        self.arrival_at: list[float] = []

    def append(self, chunk: np.ndarray) -> None:
        while self.size + len(chunk) > len(self.samples):
            self.samples = np.concatenate([self.samples, np.zeros_like(self.samples)])
        self.samples[self.size : self.size + len(chunk)] = chunk
        self.size += len(chunk)
        self.arrival_ms.append(self.size * 1000 // SAMPLE_RATE)
        self.arrival_at.append(time.monotonic())

    def slice(self, start_ms: int, end_ms: int) -> np.ndarray:
        return self.samples[start_ms * SAMPLE_RATE // 1000 : end_ms * SAMPLE_RATE // 1000].copy()

    def heard_at(self, ms: int) -> float:
        """The wall time when the audio at `ms` arrived."""
        index = bisect_left(self.arrival_ms, ms)
        return self.arrival_at[index] if index < len(self.arrival_at) else time.monotonic()


class Pipeline:
    def __init__(
        self, store: Store, models: Models, llm: LLM, config: RunConfig, *, today: str, intro_end_ms: int | None
    ) -> None:
        self.store = store
        self.models = models
        self.llm = llm
        self.config = config
        self.today = today
        self.intro_end_ms = intro_end_ms
        self.builder = TurnBuilder(config.turn_rules)
        self.audio = AudioBuffer()
        self.turns: asyncio.Queue[Turn | PromptJob | None] = asyncio.Queue()
        self.jobs: asyncio.Queue[SegmentJob | PromptJob | None] = asyncio.Queue()
        self.prompter_tasks: set[asyncio.Task] = set()
        self.voice_sums: dict[str, np.ndarray] = {}
        self.paused = True
        self.active_label: str | None = None
        self.intro_over = False

    async def run(self, chunks: AsyncIterator[np.ndarray]) -> None:
        self.models.detector.start()
        self.intro_over = self.intro_end_ms == 0
        self.store.set_phase("conversation" if self.intro_over else "intro")
        workers = [asyncio.create_task(self._transcribe_turns()), asyncio.create_task(self._extract_jobs())]
        try:
            async for chunk in chunks:
                for worker in workers:
                    if worker.done():
                        worker.result()
                self.audio.append(chunk)
                frames = await asyncio.to_thread(self.models.detector.push, chunk)
                await self._frames(frames)
            await self._frames(await asyncio.to_thread(self.models.detector.finish))
            await self._turns(self.builder.close())
            await self.turns.put(None)
            await asyncio.gather(*workers)
            await asyncio.gather(*self.prompter_tasks)
        finally:
            for worker in workers:
                worker.cancel()
        self.store.audio_ms = self.builder.now_ms
        await summarize(self.store, self.llm, model=self.config.summary_model, seed=self.config.seed)
        self.store.set_phase("ended")

    async def add_text(self, speaker: str, text: str) -> dict:
        """Adds a typed turn, for a run without audio. The speaker is a name, and a new name adds a person."""
        matches = name_matches(self.store.conn, speaker)
        person_id = matches[0]["id"] if matches else None
        if person_id is None:
            first = speaker.split()[0]
            person_id = self.store.create_person(
                name=speaker, aliases=[first] if first != speaker else [], gender="unknown"
            )["id"]
        label = f"text{person_id}"
        start_ms = max((s["end_ms"] for s in self.store.graph.segments.values()), default=0) + 500
        self.store.audio_ms = start_ms + 2000
        self.store.set_speaker(label, person_id, None)
        segment = self.store.add_segment(
            start_ms=start_ms,
            end_ms=start_ms + 2000,
            speaker_label=label,
            person_id=person_id,
            confidence=None,
            text=text,
            timings={},
        )
        await self.jobs.put(SegmentJob(segment["id"], time.monotonic()))
        return segment

    def end_intro(self) -> None:
        """Ends introductions at the current audio time, for a live session."""
        self.intro_end_ms = self.builder.now_ms

    async def _frames(self, frames: np.ndarray) -> None:
        if not self.intro_over and self.intro_end_ms is not None:
            split = max(0, (self.intro_end_ms - self.builder.now_ms) // 10)
            if split < len(frames):
                await self._turns(self.builder.push(frames[:split]))
                await self._turns(self.builder.close())
                self.intro_over = True
                self.store.audio_ms = self.builder.now_ms
                self.store.set_phase("conversation")
                await self.turns.put(PromptJob("opener", self.builder.now_ms))
                frames = frames[split:]
        await self._turns(self.builder.push(frames))
        self._activity()
        if self.intro_over and not self.paused and self.builder.silence_ms >= self.config.prompt_pause_ms:
            self.paused = True
            await self.turns.put(PromptJob("pause", self.builder.now_ms))

    def _activity(self) -> None:
        speaking = self.builder.speaker is not None and self.builder.silence_ms < ACTIVITY_SILENCE_MS
        label = f"spk{self.builder.speaker}" if speaking else None
        if label != self.active_label:
            self.active_label = label
            person_id = self.store.graph.speakers.get(label) if label else None
            self.store.notify({"type": "activity", "speaker_label": label, "person_id": person_id})

    async def _turns(self, turns: list[Turn]) -> None:
        for turn in turns:
            self.paused = False
            await self.turns.put(turn)

    async def _transcribe_turns(self) -> None:
        while (turn := await self.turns.get()) is not None:
            if isinstance(turn, PromptJob):
                await self.jobs.put(turn)
                continue
            heard_at = self.audio.heard_at(turn.end_ms)
            detect_ms = int((time.monotonic() - heard_at) * 1000)
            audio = self.audio.slice(turn.start_ms, turn.end_ms)

            started = time.monotonic()
            text = await asyncio.to_thread(self.models.transcriber.transcribe, audio)
            transcribe_ms = int((time.monotonic() - started) * 1000)
            started = time.monotonic()
            voice = await asyncio.to_thread(self.models.embedder.embed, audio)
            embed_ms = int((time.monotonic() - started) * 1000)
            if not text:
                continue

            intro = self.intro_end_ms is None or turn.start_ms < self.intro_end_ms
            label, confidence = self._identify(f"spk{turn.speaker}", voice, intro=intro)
            self.store.audio_ms = turn.end_ms
            segment = self.store.add_segment(
                start_ms=turn.start_ms,
                end_ms=turn.end_ms,
                speaker_label=label,
                person_id=self._person(label, confidence),
                confidence=confidence,
                text=text,
                timings={"detect_ms": detect_ms, "transcribe_ms": transcribe_ms, "embed_ms": embed_ms},
            )
            await self.jobs.put(SegmentJob(segment["id"], heard_at))
        await self.jobs.put(None)

    def _identify(self, label: str, voice: np.ndarray, *, intro: bool) -> tuple[str, float | None]:
        """Saves voices during introductions and returns no score. Later, confirms the detected speaker by
        voice, or picks the closest saved voice. A score under the threshold marks an unknown speaker."""
        voices = self.store.voices()
        if intro:
            self._save_voice(label, voice)
            return label, None
        scores = {lbl: float(voice @ saved) for lbl, (_, saved) in voices.items()}
        if scores.get(label, 0.0) >= self.config.voice_threshold:
            return label, scores[label]
        best = max(scores, key=scores.__getitem__, default=None)
        if best is not None and scores[best] >= self.config.voice_threshold:
            return best, scores[best]
        if label not in voices:
            self._save_voice(label, voice)
        return label, max(scores.values(), default=0.0)

    def _save_voice(self, label: str, voice: np.ndarray) -> None:
        total = self.voice_sums.get(label, np.zeros_like(voice)) + voice
        self.voice_sums[label] = total
        self.store.set_speaker(label, self.store.graph.speakers.get(label), total / np.linalg.norm(total))

    def _person(self, label: str, confidence: float | None) -> int | None:
        if confidence is not None and confidence < self.config.voice_threshold:
            return None
        return self.store.graph.speakers.get(label)

    async def _extract_jobs(self) -> None:
        while (job := await self.jobs.get()) is not None:
            match job:
                case SegmentJob():
                    await self._extract(job)
                case PromptJob():
                    call = prompter_call(
                        self.store,
                        job.trigger,
                        until_ms=job.at_ms,
                        model=self.config.extract_model,
                        seed=self.config.seed,
                    )
                    if call is not None:
                        task = asyncio.create_task(suggest(self.store, self.llm, call, at_ms=job.at_ms))
                        self.prompter_tasks.add(task)

    async def _extract(self, job: SegmentJob) -> None:
        segment = self.store.graph.segments[job.segment_id]
        speaker_id = self._person(segment["speaker_label"], segment["speaker_confidence"])
        call = extraction_call(
            self.store,
            segment=segment,
            speaker_id=speaker_id,
            today=self.today,
            model=self.config.extract_model,
            seed=self.config.seed,
            context=self.config.context,
        )
        started = time.monotonic()
        result = await self.llm.call(call, run_id=self.store.run_id, audio_ms=segment["end_ms"])
        extract_ms = int((time.monotonic() - started) * 1000)

        self.store.audio_ms = segment["end_ms"]
        output = result.output
        if output is not None:
            teller = Teller(segment["id"], speaker_id, segment["speaker_confidence"] or 1.0, result.model_call_id)
            name = said_name(segment["text"], output["speaker_name"]) if speaker_id is None else None
            if name:
                speaker_id = self._name_speaker(segment, name, Held(output, teller))
            if speaker_id is not None or not name:
                apply_output(self.store, Held(output, replace(teller, person_id=speaker_id)))
        total_ms = int((time.monotonic() - job.heard_at) * 1000)
        self.store.update_segment(segment["id"], timings={"extract_ms": extract_ms, "total_ms": total_ms})

    def _name_speaker(self, segment: dict, name: str, held: Held) -> int | None:
        """Links a voice to the person who said their name. Asks the room when the name fits several people."""
        entry = next((p for p in held.output["people"] if p["name"].lower() == name.lower()), None)
        matches = name_matches(self.store.conn, name)
        if len(matches) > 1:
            show_prompt(
                self.store,
                text=f"Which {name} is speaking: {' or '.join(m['name'] for m in matches)}? Or add a nickname.",
                reason="which_speaker",
                person_ids=[m["id"] for m in matches],
                payload={"label": segment["speaker_label"], "name": name, "held": held_payload(held)},
                model_call_id=held.teller.model_call_id,
            )
            return None
        if matches:
            person_id = matches[0]["id"]
        else:
            first = name.split()[0]
            person_id = self.store.create_person(
                name=name, aliases=[first] if first != name else [], gender=entry["gender"] if entry else "unknown"
            )["id"]
        link_speaker(self.store, segment["speaker_label"], person_id, threshold=self.config.voice_threshold)
        return person_id


def link_speaker(store: Store, label: str, person_id: int, *, threshold: float) -> None:
    """Names a voice, and the earlier turns that the voice said."""
    with store.batch():
        store.set_speaker(label, person_id, None)
        for segment in list(store.graph.segments.values()):
            confidence = segment["speaker_confidence"]
            if (
                segment["speaker_label"] == label
                and segment["person_id"] is None
                and (confidence is None or confidence >= threshold)
            ):
                store.update_segment(segment["id"], person_id=person_id)
