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
from grapevine.extract import SYSTEM as EXTRACT_SYSTEM
from grapevine.extract import Held, Teller, apply_output, extraction_call, held_payload, read_again, respell_people
from grapevine.llm import LLM, watch_on
from grapevine.models import Hints, Models
from grapevine.names import english, misspelled, proper_names, spell_names
from grapevine.prompter import prompter_call, show_prompt, suggest
from grapevine.review import RELATIONSHIPS_SYSTEM, review
from grapevine.store import Store, name_matches
from grapevine.summarize import summarize
from grapevine.turns import Turn, TurnBuilder

log = logging.getLogger(__name__)

ACTIVITY_SILENCE_MS = 300
PARTIAL_MS = 300
MIN_PARTIAL_MS = 500
# Each preview costs about 155 ms of GPU, which the language model also needs, so a preview waits for new words.
NEW_AUDIO_MS = 700
# A preview holds Whisper for up to this long, so none starts when the open turn may end before it is done.
PREVIEW_MS = 300
MAX_PARTIAL_MS = 15000
SPEECH_RMS = 0.01
VOICE_TIE = 0.02
MIN_VOICE_MS = 2000
# A turn under MIN_VOICE_MS by the same speaker still scores 0.8 or more against their voice.
SHORT_UNLIKE = 0.65
SETTLE_MARGIN = 0.05
# A turn this far from every person's voice keeps the speaker it has.
SETTLE_FLOOR = 0.7
MAX_VOICES = 100
# Whisper reads at most 224 prompt tokens, so the hints keep to about 40 terms and the end of the last lines.
MAX_TERMS = 40
RECENT_CHARS = 200
LEADING_NAME = re.compile(r"(?:(?:Hi|Hey|Hello|So),?\s+)?([A-Z][a-z]+ [A-Z][a-z]+)(?:[,.]|$)")
INTRODUCES = re.compile(r"\bthis is (?:my [a-z]+)?\s*([A-Z][a-z]+)?[.!]?$")
BARE_NAME = re.compile(r"([A-Z][a-z]+)[.!,]?(?:\s|$)")
SELF_INTRO = re.compile(r"\b(?:I'm|I am|my name is|name's|call me)\s+([A-Z][a-z]+)\b(?!['’]s)")


class RunStopping(Exception):
    """A typed turn came after the run stopped taking turns, so no worker would read it."""


def goes_by(person: dict, name: str) -> bool:
    first = name.split()[0].lower()
    return first in {person["name"].split()[0].lower(), *(a.lower() for a in person["aliases"])}


def said_name(
    text: str, model_name: str | None, *, intro: bool, unnamed: bool, roster: list[str], introduced: str | None = None
) -> str | None:
    """The name speakers give for themselves. The model's name counts only when the turn says it. Outside
    introductions, the turn must also say "I'm" or "my name is" before the name. A full name at the start
    of a turn, as in "Chris Porter, I'm an accountant", counts only for an unnamed voice in introductions,
    and only when it is on the roster if there is one, so a reply like "Most Sundays." names no one.

    `introduced` is set when another voice just ended a turn with "this is my brother" or "this is Greta",
    and holds the name it gave, if any. Then an unnamed voice in introductions that starts with a bare first
    name, as in "Felix.", is that person, and one that gives no name is the person just introduced."""
    first_names = [m[1] for m in SELF_INTRO.finditer(text)]
    if intro and model_name and all(word in text for word in model_name.split()):
        return model_name
    leading = LEADING_NAME.match(text)
    if intro and unnamed and leading and (not roster or leading[1] in roster):
        return leading[1]
    if intro and unnamed and introduced is not None and not first_names and not leading:
        bare = BARE_NAME.match(text)
        firsts = {name.split()[0] for name in roster}
        if bare and (bare[1] in firsts if roster else bare[1] in proper_names() and bare[1].lower() not in english()):
            return bare[1]
        if introduced and (not roster or introduced in firsts):
            return introduced
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


@dataclass(frozen=True)
class RereadJob:
    segment_ids: list[int]


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
        self,
        store: Store,
        models: Models,
        llm: LLM,
        config: RunConfig,
        *,
        today: str,
        intro_end_ms: int | None,
        roster: list[str],
        vocabulary: list[str],
    ) -> None:
        self.store = store
        self.models = models
        self.llm = llm
        self.config = config
        self.today = today
        self.intro_end_ms = intro_end_ms
        self.roster = roster
        self.vocabulary = vocabulary
        self.builder = TurnBuilder(config.turn_rules)
        self.audio = AudioBuffer()
        self.turns: asyncio.Queue[Turn | PromptJob | None] = asyncio.Queue()
        self.jobs: asyncio.Queue[SegmentJob | PromptJob | RereadJob | None] = asyncio.Queue()
        self.prompter_tasks: set[asyncio.Task] = set()
        self.voice_sums: dict[str, np.ndarray] = {}
        self.voice_ms: dict[str, int] = {}
        self.segment_voices: dict[int, tuple[np.ndarray, int]] = {}
        self.extracted_as: dict[int, int | None] = {}
        # Turns whose text the room edited, read again even when their speaker did not change.
        self.edited: set[int] = set()
        self.paused = True
        self.active_label: str | None = None
        self.intro_over = False
        self.partials = False
        self.closed_ms = 0
        self.transcribed_ms = 0
        self.previewed_ms = 0
        self.whisper = asyncio.Lock()
        self.fed = False

    async def run(self, chunks: AsyncIterator[np.ndarray]) -> None:
        self.llm.warm(self.config.extract_model, EXTRACT_SYSTEM)
        self.models.detector.start()
        self.intro_over = self.intro_end_ms == 0
        self.store.set_phase("conversation" if self.intro_over else "intro")
        self._seat_roster()
        workers = [asyncio.create_task(self._transcribe_turns()), asyncio.create_task(self._extract_jobs())]
        partials = asyncio.create_task(self._partials()) if self.partials else None
        feed = asyncio.create_task(self._feed(chunks))
        try:
            # A typed run gets no chunks, so a failed worker must end the run without waiting for one.
            await asyncio.wait([feed, *workers], return_when=asyncio.FIRST_EXCEPTION)
            await asyncio.gather(feed, *workers)
            await asyncio.gather(*self.prompter_tasks)
        finally:
            for task in [feed, *workers, partials]:
                if task is not None:
                    task.cancel()
        self.store.audio_ms = self.builder.now_ms
        self.store.set_phase("review")
        self._settle_speakers()
        self.llm.warm(self.config.summary_model, RELATIONSHIPS_SYSTEM)
        await self._reextract(list(self.extracted_as))
        await review(self.store, self.llm, model=self.config.summary_model, seed=self.config.seed, today=self.today)
        await summarize(self.store, self.llm, model=self.config.summary_model, seed=self.config.seed)
        # The extractor has stopped, so a fix that the room made during the review waits here as a RereadJob.
        while not self.jobs.empty():
            await self._reextract(self.jobs.get_nowait().segment_ids)
        self.store.set_phase("ended")

    async def _feed(self, chunks: AsyncIterator[np.ndarray]) -> None:
        async for chunk in chunks:
            self.audio.append(chunk)
            frames = await asyncio.to_thread(self.models.detector.push, chunk)
            await self._frames(frames)
        self.fed = True
        await self._frames(await asyncio.to_thread(self.models.detector.finish))
        await self._turns(self.builder.close())
        await self.turns.put(None)

    def _seat_roster(self) -> None:
        """Puts the people the room typed in before the session into the graph. A typed name that one known
        person has keeps that person, so a session adds to what earlier sessions said about them."""
        with self.store.batch():
            ids = []
            for name in self.roster:
                matches = [p for p in name_matches(self.store.conn, name) if p["name"] == name]
                if len(matches) == 1:
                    self.store.show_person(matches[0]["id"])
                    ids.append(matches[0]["id"])
                else:
                    first = name.split()[0]
                    aliases = [first] if first != name else []
                    ids.append(self.store.create_person(name=name, aliases=aliases, gender="unknown")["id"])
            if ids:
                self.store.emit({"type": "roster", "person_ids": ids})

    async def add_text(self, speaker: str, text: str) -> dict:
        """Adds a typed turn, for a run without audio. The speaker is a name, and a new name adds a person."""
        if self.fed:
            raise RunStopping("the run is stopping and takes no more turns")
        matches = self._roster_matches(speaker) or name_matches(self.store.conn, speaker)
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

    def reread(self, segment_ids: list[int], *, edited: bool = False) -> None:
        """Extracts turns again after the room fixed their speaker or their text by hand."""
        if edited:
            self.edited.update(segment_ids)
        self.jobs.put_nowait(RereadJob(segment_ids))

    def rename(self, old: str, new: str) -> None:
        """Keeps the roster, which spells names in the transcript, in step with a person the room renamed."""
        self.roster = [new if name == old else name for name in self.roster]

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
            self.closed_ms = turn.end_ms
            self.paused = False
            await self.turns.put(turn)

    def _hints(self) -> Hints:
        """The roster, the words the room gave, and the names of people, groups, and events this run has
        learned, in that order so a long list keeps the first two, with the end of the last two lines."""
        graph = self.store.graph
        learned = [p["name"] for p in graph.people.values() if p["merged_into_id"] is None]
        learned += [g["name"] for g in graph.groups.values()] + [e["name"] for e in graph.events.values()]
        terms = list(dict.fromkeys([*self.roster, *self.vocabulary, *learned]))[:MAX_TERMS]
        last = sorted(graph.segments.values(), key=lambda s: s["start_ms"])[-2:]
        return Hints(terms, " ".join(s["text"] for s in last)[-RECENT_CHARS:])

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
                audio = self.audio.slice(start_ms, end_ms)
                async with self.whisper:
                    text = await asyncio.to_thread(self.models.transcriber.transcribe, audio, self._hints())
                text = spell_names(text, self.roster)
                transcribe_ms = int((time.monotonic() - started) * 1000)
                if not text:
                    continue
                intro = self.intro_end_ms is None or start_ms < self.intro_end_ms
                label, confidence = self._identify(f"spk{turn.speaker}", voice, ms=end_ms - start_ms, intro=intro)
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
                self.segment_voices[segment["id"]] = (voice, end_ms - start_ms)
                await self.jobs.put(SegmentJob(segment["id"], heard_at))
            self.transcribed_ms = turn.end_ms
            self.store.notify({"type": "partial", "start_ms": turn.start_ms, "text": None})
        await self.jobs.put(None)

    async def _partials(self) -> None:
        """Shows the words of speech that no turn holds yet, for a run that keeps pace with real time. The
        speaker detector opens a turn more than a second after speech starts, so loud audio after the last
        turn starts a preview before that. The text is only a preview: it is not saved. No preview starts while a
        closed turn waits for Whisper, or when the open turn may close before the preview is done, so the final
        transcription of a turn does not wait for one."""
        while True:
            await asyncio.sleep(PARTIAL_MS / 1000)
            speaker = self.builder.speaker
            end_ms = self.audio.size * 1000 // SAMPLE_RATE
            start_ms = max(self.builder.start_ms if speaker is not None else self.closed_ms, end_ms - MAX_PARTIAL_MS)
            waiting = self.closed_ms > self.transcribed_ms
            closing = (
                speaker is not None and self.builder.silence_ms + PREVIEW_MS >= self.config.turn_rules.end_silence_ms
            )
            if self.whisper.locked() or waiting or closing:
                continue
            if end_ms - start_ms < MIN_PARTIAL_MS or end_ms - self.previewed_ms < NEW_AUDIO_MS:
                continue
            recent = self.audio.slice(max(start_ms, end_ms - 1000), end_ms)
            if speaker is None and float(np.sqrt(np.mean(recent**2))) < SPEECH_RMS:
                continue
            self.previewed_ms = end_ms
            async with self.whisper:
                audio = self.audio.slice(start_ms, end_ms)
                text = await asyncio.to_thread(self.models.transcriber.transcribe, audio, self._hints())
            if not text or self.closed_ms > start_ms:
                continue
            label = f"spk{speaker}" if speaker is not None else None
            self.store.notify(
                {
                    "type": "partial",
                    "start_ms": start_ms,
                    "speaker_label": label,
                    "person_id": self.store.graph.speakers.get(label) if label else None,
                    "text": spell_names(text, self.roster),
                }
            )

    def _pieces(self, turn: Turn) -> list[tuple[int, int, np.ndarray]]:
        """Splits a turn at its pauses where the voice changes, since the speaker detector can give two
        similar voices one label. Each piece comes with its voice. A piece shorter than MIN_VOICE_MS joins
        the piece before it, because the voice of a short piece is not reliable."""
        bounds = list(pairwise([turn.start_ms, *turn.pauses, turn.end_ms]))
        voiced = [bound for i, bound in enumerate(bounds) if i == 0 or bound[1] - bound[0] >= MIN_VOICE_MS]
        clips = [self.audio.slice(start_ms, end_ms) for start_ms, end_ms in voiced]
        voices = dict(zip(voiced, self.models.embedder.embed(clips), strict=True))
        pieces: list[tuple[int, int, np.ndarray]] = []
        for start_ms, end_ms in bounds:
            if pieces:
                last_start, _, last_voice = pieces[-1]
                if (start_ms, end_ms) not in voices:
                    pieces[-1] = (last_start, end_ms, last_voice)
                    continue
                if float(voices[start_ms, end_ms] @ last_voice) >= self.config.new_voice_threshold:
                    joined = self.models.embedder.embed([self.audio.slice(last_start, end_ms)])[0]
                    pieces[-1] = (last_start, end_ms, joined)
                    continue
            pieces.append((start_ms, end_ms, voices[start_ms, end_ms]))
        return pieces

    def _identify(self, label: str, voice: np.ndarray, *, ms: int, intro: bool) -> tuple[str, float | None]:
        """Picks the voice label of a turn. The voice decides, and the detected speaker only breaks near ties,
        because the speaker detector can merge two similar voices into one speaker.

        During introductions it saves voices and returns no score. A detected speaker whose new turn sounds
        unlike its saved voice is two people, so the turn gets its own label. Only turns and saved voices of
        at least MIN_VOICE_MS count for that, because short audio gives noisy voices. Later, a score under
        the threshold marks an unknown speaker."""
        voices = self.store.voices()
        scores = {lbl: float(voice @ saved) for lbl, (_, saved) in voices.items()}
        if intro:
            reliable = ms >= MIN_VOICE_MS and self.voice_ms.get(label, 0) >= MIN_VOICE_MS
            if reliable and scores[label] < self.config.new_voice_threshold:
                best = max(scores, key=scores.__getitem__)
                label = best if scores[best] >= self.config.new_voice_threshold else self._new_label(label, voices)
            if ms >= MIN_VOICE_MS or label not in voices:
                self._save_voice(label, voice, ms)
            return label, None

        best = max(scores, key=scores.__getitem__, default=None)
        if best is None:
            self._save_voice(label, voice, ms)
            return label, 0.0
        chosen = label if label in scores and scores[label] >= scores[best] - VOICE_TIE else best
        if scores[chosen] < self.config.voice_threshold and label not in voices:
            self._save_voice(label, voice, ms)
            return label, scores[chosen]
        return chosen, scores[chosen]

    def _new_label(self, label: str, voices: dict) -> str:
        return next(f"{label}+{n}" for n in range(1, MAX_VOICES) if f"{label}+{n}" not in voices)

    def _settle_intro_voices(self) -> None:
        """Gives a voice that no one named during introductions to the closest named voice, when it is close
        enough and clearly closer than any other. Such a voice is usually the second half of a long
        introduction."""
        voices = self.store.voices()
        named = {lbl: saved for lbl, (pid, saved) in voices.items() if pid is not None}
        for label, (person_id, saved) in voices.items():
            if person_id is not None or not named:
                continue
            ranked = sorted((float(saved @ voice), lbl) for lbl, voice in named.items())[::-1]
            (score, best), runner_up = ranked[0], (ranked[1][0] if len(ranked) > 1 else 0.0)
            if score >= self.config.voice_threshold and score - runner_up >= SETTLE_MARGIN:
                link_speaker(self.store, label, voices[best][0], threshold=self.config.voice_threshold)

    def _save_voice(self, label: str, voice: np.ndarray, ms: int) -> None:
        """Keeps the mean voice of a label, weighted by the length of each turn."""
        total = self.voice_sums.get(label, np.zeros_like(voice)) + voice * ms
        self.voice_sums[label] = total
        self.voice_ms[label] = self.voice_ms.get(label, 0) + ms
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
                case RereadJob():
                    await self._reextract(job.segment_ids)
                case PromptJob():
                    self.store.audio_ms = job.at_ms
                    if job.trigger == "opener":
                        self._settle_intro_voices()
                    else:
                        self._settle_speakers()
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
            await self._reextract(list(self.extracted_as))

    async def _extract(self, job: SegmentJob) -> None:
        segment = self.store.graph.segments[job.segment_id]
        respell_people(self.store, segment["text"])
        manual = segment["speaker_manual"]
        speaker_id = (
            segment["person_id"] if manual else self._person(segment["speaker_label"], segment["speaker_confidence"])
        )
        call = extraction_call(
            self.store,
            segment=segment,
            speaker_id=speaker_id,
            today=self.today,
            model=self.config.extract_model,
            seed=self.config.seed,
            context=self.config.context,
        )
        self.extracted_as[segment["id"]] = speaker_id
        started = time.monotonic()
        watch = watch_on(self.store, worker="extractor", segment_id=segment["id"])
        result = await self.llm.call(call, run_id=self.store.run_id, audio_ms=segment["end_ms"], watch=watch)
        extract_ms = int((time.monotonic() - started) * 1000)

        self.store.audio_ms = segment["end_ms"]
        output = result.output
        if output is not None:
            teller = Teller(segment["id"], speaker_id, segment["speaker_confidence"] or 1.0, result.model_call_id)
            intro = segment["speaker_confidence"] is None
            name = None
            if not manual and (speaker_id is None or intro):
                name = said_name(
                    segment["text"],
                    output["speaker_name"],
                    intro=intro,
                    unnamed=speaker_id is None,
                    roster=self.roster,
                    introduced=self._introduced(segment),
                )
            if name and speaker_id is not None and goes_by(self.store.person(speaker_id), name):
                name = None
            voiced = segment["id"] in self.segment_voices
            if name and voiced and (speaker_id is not None or self._sounds_unlike_label(segment)):
                segment = self._own_voice(segment)
            if name:
                speaker_id = self._name_speaker(segment, name, Held(output, teller))
                if speaker_id is not None and segment["person_id"] != speaker_id:
                    self.store.update_segment(segment["id"], person_id=speaker_id)
            # A read under another speaker tells facts such as "my name is Mia, not Dana", so it is read again.
            moved = teller.person_id not in (None, speaker_id)
            if not moved and (speaker_id is not None or not name):
                apply_output(self.store, Held(output, replace(teller, person_id=speaker_id)))
        total_ms = int((time.monotonic() - job.heard_at) * 1000)
        self.store.update_segment(segment["id"], timings={"extract_ms": extract_ms, "total_ms": total_ms})

    def _introduced(self, segment: dict) -> str | None:
        """The name that the turn before, by another voice, gave when it ended with "this is ...", or "" when it
        gave none. None when that turn introduced no one."""
        before = [
            s
            for s in self.store.graph.segments.values()
            if s["start_ms"] < segment["start_ms"] and s["speaker_label"] != segment["speaker_label"]
        ]
        if not before:
            return None
        cue = INTRODUCES.search(max(before, key=lambda s: s["start_ms"])["text"])
        return (cue[1] or "") if cue else None

    def _settle_speakers(self) -> None:
        """Gives each turn to the person whose voice is closest. A person's voice here is the mean of all their
        long turns so far, which is steadier than the voice of one introduction. Turns the voice detector got
        wrong, or left unknown, move to the right person. A turn moves only when its closest voice beats its
        current person, or the next closest voice, by SETTLE_MARGIN, because the voice of a short turn is noisy.
        An unknown turn, or a turn of a person with no voice here, moves only past the voice threshold, because
        it may belong to a person with no voice yet, unless its closest voice is the person its label already
        names. A wrong speaker is worse than an unknown one, because the room can fix an unknown one with one
        click."""
        sums: dict[int, np.ndarray] = {}
        for segment_id, (voice, ms) in self.segment_voices.items():
            person_id = self.store.graph.segments[segment_id]["person_id"]
            if person_id is not None and ms >= MIN_VOICE_MS:
                sums[person_id] = sums.get(person_id, 0) + voice * ms
        if len(sums) < 2:
            return
        people = {pid: total / np.linalg.norm(total) for pid, total in sums.items()}
        with self.store.batch():
            for segment_id, (voice, _) in self.segment_voices.items():
                segment = self.store.graph.segments[segment_id]
                scores = {pid: float(voice @ v) for pid, v in people.items()}
                person_id = max(scores, key=scores.__getitem__)
                score = scores.pop(person_id)
                rival = scores.get(segment["person_id"], max(scores.values()))
                agrees = person_id == self.store.graph.speakers.get(segment["speaker_label"])
                floor = SETTLE_FLOOR if segment["person_id"] in people or agrees else self.config.voice_threshold
                clear = score >= floor and score - rival >= SETTLE_MARGIN
                if clear and person_id != segment["person_id"] and not segment["speaker_manual"]:
                    self.store.update_segment(segment_id, person_id=person_id, speaker_confidence=score)

    async def _reextract(self, segment_ids: list[int]) -> None:
        """Extracts again each turn whose speaker changed after its extraction, because its facts name the
        wrong teller, and each turn whose text the room edited. The new facts revise the old ones."""
        for segment_id in segment_ids:
            segment = self.store.graph.segments[segment_id]
            if segment_id not in self.extracted_as:
                continue
            edited = segment_id in self.edited
            if not edited and segment["person_id"] in (None, self.extracted_as[segment_id]):
                continue
            # A fix during the read marks the turn again, and its RereadJob reads the new text.
            self.edited.discard(segment_id)
            answered = await read_again(
                self.store,
                self.llm,
                segment_id,
                today=self.today,
                model=self.config.extract_model,
                seed=self.config.seed,
                context=self.config.context,
            )
            if answered:
                self.extracted_as[segment_id] = segment["person_id"]
            elif edited:
                self.edited.add(segment_id)

    def _sounds_unlike_label(self, segment: dict) -> bool:
        """Whether a turn sounds unlike the voice its label saved from other turns. The voice of a short turn
        is noisy, so it must score far lower to count."""
        label = segment["speaker_label"]
        voice, ms = self.segment_voices[segment["id"]]
        saved = self.store.voices().get(label)
        others_ms = self.voice_ms.get(label, 0) - (ms if ms >= MIN_VOICE_MS else 0)
        bar = self.config.voice_threshold if ms >= MIN_VOICE_MS else SHORT_UNLIKE
        return saved is not None and others_ms > 0 and float(voice @ saved[1]) < bar

    def _own_voice(self, segment: dict) -> dict:
        """Moves an introduction to a voice of its own. Someone said a new name under a voice that already
        has a name, or that sounds like someone else, so the speaker detector merged two people here."""
        label = self._new_label(segment["speaker_label"], self.store.voices())
        voice, ms = self.segment_voices[segment["id"]]
        self.voice_sums[label] = voice * ms
        self.voice_ms[label] = ms
        self.store.set_speaker(label, None, voice)
        self.store.update_segment(segment["id"], speaker_label=label)
        return self.store.graph.segments[segment["id"]]

    def _name_speaker(self, segment: dict, name: str, held: Held) -> int | None:
        """Links a voice to the person who said their name. Asks the room when the name fits several people and
        is the exact name of none or several of them. A full name said this way respells a known person, because
        people spell their own names best."""
        entry = next((p for p in held.output["people"] if p["name"].lower() == name.lower()), None)
        matches = self._roster_matches(name) or name_matches(self.store.conn, name)
        matches = [m for m in matches if m["name"].lower() == name.lower()] or matches
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
            old = matches[0]["name"]
            if " " in name and old.lower() != name.lower() and person_id not in self.store.graph.roster:
                self.store.update_person(person_id, add_aliases=[old], gender=None, name=name)
        else:
            first = name.split()[0]
            person_id = self.store.create_person(
                name=name, aliases=[first] if first != name else [], gender=(entry or {}).get("gender") or "unknown"
            )["id"]
        link_speaker(self.store, segment["speaker_label"], person_id, threshold=self.config.voice_threshold)
        return person_id

    def _roster_matches(self, name: str) -> list[dict]:
        """Roster people a self-introduction could mean: by first name or alias for one word, else by the
        full name, spelled right or nearly so."""
        people = [self.store.graph.people[pid] for pid in self.store.graph.roster]
        if " " not in name:
            return [p for p in people if goes_by(p, name)]
        exact = [p for p in people if p["name"].lower() == name.lower()]
        return exact or [p for p in people if misspelled(p["name"], name)]


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
