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
from itertools import pairwise

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
VOICE_TIE = 0.02
MIN_VOICE_MS = 1000
MAX_VOICES = 100
SELF_INTRO = re.compile(r"\b(?:I'm|I am|my name is|name's|call me)\s+([A-Z][a-z]+)")


def said_name(text: str, model_name: str | None, *, intro: bool) -> str | None:
    """The name speakers give for themselves. The model's name counts only when the turn says it. Outside
    introductions, the turn must also say "I'm" or "my name is" before the name."""
    first_names = [m[1] for m in SELF_INTRO.finditer(text)]
    if intro and model_name and all(word in text for word in model_name.split()):
        return model_name
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

            started = time.monotonic()
            pieces = await asyncio.to_thread(self._pieces, turn)
            embed_ms = int((time.monotonic() - started) * 1000)
            for start_ms, end_ms, voice in pieces:
                started = time.monotonic()
                text = await asyncio.to_thread(self.models.transcriber.transcribe, self.audio.slice(start_ms, end_ms))
                transcribe_ms = int((time.monotonic() - started) * 1000)
                if not text:
                    continue
                intro = self.intro_end_ms is None or start_ms < self.intro_end_ms
                label, confidence = self._identify(f"spk{turn.speaker}", voice, intro=intro)
                self.store.audio_ms = end_ms
                segment = self.store.add_segment(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    speaker_label=label,
                    person_id=self._person(label, confidence),
                    confidence=confidence,
                    text=text,
                    timings={"detect_ms": detect_ms, "transcribe_ms": transcribe_ms, "embed_ms": embed_ms},
                )
                await self.jobs.put(SegmentJob(segment["id"], heard_at))
        await self.jobs.put(None)

    def _pieces(self, turn: Turn) -> list[tuple[int, int, np.ndarray]]:
        """Splits a turn at its pauses where the voice changes, since the speaker detector can give two
        similar voices one label. Each piece comes with its voice."""
        bounds = [turn.start_ms, *turn.pauses, turn.end_ms]
        pieces: list[tuple[int, int, np.ndarray]] = []
        for start_ms, end_ms in pairwise(bounds):
            voice = self.models.embedder.embed(self.audio.slice(start_ms, end_ms))
            if pieces:
                last_start, _, last_voice = pieces[-1]
                short = end_ms - start_ms < MIN_VOICE_MS
                if short or float(voice @ last_voice) >= self.config.new_voice_threshold:
                    joined = (
                        self.models.embedder.embed(self.audio.slice(last_start, end_ms)) if not short else last_voice
                    )
                    pieces[-1] = (last_start, end_ms, joined)
                    continue
            pieces.append((start_ms, end_ms, voice))
        return pieces

    def _identify(self, label: str, voice: np.ndarray, *, intro: bool) -> tuple[str, float | None]:
        """Picks the voice label of a turn. The voice decides, and the detected speaker only breaks near ties,
        because the speaker detector can merge two similar voices into one speaker.

        During introductions it saves voices and returns no score. A detected speaker whose new turn sounds
        unlike its saved voice is two people, so the turn gets its own label. Later, a score under the
        threshold marks an unknown speaker."""
        voices = self.store.voices()
        scores = {lbl: float(voice @ saved) for lbl, (_, saved) in voices.items()}
        if intro:
            if label in scores and scores[label] < self.config.new_voice_threshold:
                best = max(scores, key=scores.__getitem__)
                label = best if scores[best] >= self.config.new_voice_threshold else self._new_label(label, voices)
            self._save_voice(label, voice)
            return label, None

        best = max(scores, key=scores.__getitem__, default=None)
        if best is None:
            self._save_voice(label, voice)
            return label, 0.0
        chosen = label if label in scores and scores[label] >= scores[best] - VOICE_TIE else best
        if scores[chosen] < self.config.voice_threshold and label not in voices:
            self._save_voice(label, voice)
            return label, scores[chosen]
        return chosen, scores[chosen]

    def _new_label(self, label: str, voices: dict) -> str:
        return next(f"{label}+{n}" for n in range(1, MAX_VOICES) if f"{label}+{n}" not in voices)

    def _settle_intro_voices(self) -> None:
        """Gives a voice that no one named during introductions to the closest named voice, when it is close
        enough. Such a voice is usually the second half of a long introduction."""
        voices = self.store.voices()
        named = {lbl: saved for lbl, (pid, saved) in voices.items() if pid is not None}
        for label, (person_id, saved) in voices.items():
            if person_id is not None or not named:
                continue
            best = max(named, key=lambda lbl: float(saved @ named[lbl]))
            if float(saved @ named[best]) >= self.config.voice_threshold:
                link_speaker(self.store, label, voices[best][0], threshold=self.config.voice_threshold)

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
                    if job.trigger == "opener":
                        self.store.audio_ms = job.at_ms
                        self._settle_intro_voices()
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
            intro = segment["speaker_confidence"] is None
            name = said_name(segment["text"], output["speaker_name"], intro=intro) if speaker_id is None else None
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
