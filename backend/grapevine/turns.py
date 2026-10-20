"""Cuts speaker turns from frame-level speech probabilities."""

from dataclasses import dataclass

import numpy as np

from grapevine.models import FRAME_MS


@dataclass(frozen=True)
class TurnRules:
    """When a turn starts and ends. Tests and replays change these to compare runs."""

    speech_threshold: float = 0.5
    end_silence_ms: int = 800
    switch_ms: int = 200
    min_turn_ms: int = 400
    max_turn_ms: int = 15000
    split_pause_ms: int = 250


@dataclass(frozen=True)
class Turn:
    speaker: int
    start_ms: int
    end_ms: int


class TurnBuilder:
    """Feeds on frames in order. A turn ends when its speaker stops for `end_silence_ms`, when another
    speaker holds the floor for `switch_ms`, or at a pause once it is longer than `max_turn_ms`."""

    def __init__(self, rules: TurnRules) -> None:
        self.rules = rules
        self.now_ms = 0
        self.speaker: int | None = None
        self.start_ms = 0
        self.voice_end_ms = 0
        self.last_pause_ms: int | None = None
        self.challenger: int | None = None
        self.challenger_start_ms = 0
        self.last_speech_ms = 0

    @property
    def silence_ms(self) -> int:
        return self.now_ms - self.last_speech_ms

    def push(self, frames: np.ndarray) -> list[Turn]:
        turns: list[Turn] = []
        for probs in frames:
            ms = self.now_ms
            self.now_ms += FRAME_MS
            top = int(np.argmax(probs))
            speaking = top if probs[top] > self.rules.speech_threshold else None
            if speaking is not None:
                self.last_speech_ms = self.now_ms
            turns += self._frame(ms, speaking)
        return turns

    def _frame(self, ms: int, speaking: int | None) -> list[Turn]:
        if self.speaker is None:
            if speaking is not None:
                self._open(speaking, ms)
            return []

        if speaking == self.speaker:
            gap = ms - self.voice_end_ms
            if gap >= self.rules.split_pause_ms:
                self.last_pause_ms = self.voice_end_ms + gap // 2
            self.voice_end_ms = ms + FRAME_MS
            self.challenger = None
            if self.last_pause_ms and ms - self.start_ms > self.rules.max_turn_ms:
                cut = self.last_pause_ms
                turn = Turn(self.speaker, self.start_ms, cut)
                self._open(self.speaker, cut)
                self.voice_end_ms = ms + FRAME_MS
                return [turn]
            return []

        if speaking is None:
            self.challenger = None
            if ms - self.voice_end_ms >= self.rules.end_silence_ms:
                return self.close()
            return []

        if speaking != self.challenger:
            self.challenger = speaking
            self.challenger_start_ms = ms
        if ms + FRAME_MS - self.challenger_start_ms < self.rules.switch_ms:
            return []
        turns = self._end(min(self.voice_end_ms, self.challenger_start_ms))
        self._open(speaking, self.challenger_start_ms)
        self.voice_end_ms = ms + FRAME_MS
        return turns

    def _open(self, speaker: int, ms: int) -> None:
        self.speaker = speaker
        self.start_ms = ms
        self.voice_end_ms = ms + FRAME_MS
        self.last_pause_ms = None
        self.challenger = None

    def _end(self, end_ms: int) -> list[Turn]:
        turn = Turn(self.speaker, self.start_ms, end_ms)
        self.speaker = None
        if turn.end_ms - turn.start_ms < self.rules.min_turn_ms:
            return []
        return [turn]

    def close(self) -> list[Turn]:
        """Ends the open turn at its last speech, for example at the end of the audio or of introductions."""
        if self.speaker is None:
            return []
        return self._end(self.voice_end_ms)
