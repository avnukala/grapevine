"""The audio and language models behind small interfaces, so a test can swap each one for a fake."""

import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from grapevine.audio import SAMPLE_RATE

FRAME_MS = 10
MAX_SPEAKERS = 8
MIN_EMBED_SAMPLES = SAMPLE_RATE // 2
# The longest wait for the next piece of an answer from a server that works. A review call can wait up to 180 s
# for a free slot behind other review calls, and a cold model can take 50 s to read a prompt.
CHAT_TIMEOUT_SECONDS = 300


class SpeakerDetector(Protocol):
    def start(self) -> None:
        """Forgets all speakers before a new run."""

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Takes the next audio and returns the speech probabilities of the frames it could score, (n, 8)."""

    def finish(self) -> np.ndarray:
        """Scores the frames left at the end of the audio."""


class VoiceEmbedder(Protocol):
    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Returns a unit-length voice embedding."""


@dataclass(frozen=True)
class Hints:
    """What steers how Whisper spells a turn: names and words the group uses, and the last lines said."""

    terms: list[str]
    recent: str


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray, hints: Hints) -> str: ...


class ChatModel(Protocol):
    async def complete(self, request: dict, on_text: Callable[[str], None] | None = None) -> str:
        """Sends an OpenAI chat completion request and returns the message content. `on_text` gets the content
        so far each time more of it arrives."""


@dataclass
class Models:
    detector: SpeakerDetector
    embedder: VoiceEmbedder
    transcriber: Transcriber
    chat: ChatModel


# PyTorch on MPS breaks when two threads encode GPU work at once, and the detector and the embedder run in
# separate worker threads. Every PyTorch step on the device, copies included, holds this lock.
TORCH_GPU = threading.Lock()


class NemotronDetector:
    """Streams audio through NVIDIA Nemotron 3 Diarization. It numbers speakers in the order they first talk."""

    def __init__(self, model_id: str, *, mode: str, device: str) -> None:
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.set_streaming_mode(mode)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to(device).eval()
        self.start()

    def start(self) -> None:
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0
        self.cache = None
        self.mel_frame = 0
        self.next_start: int | None = None

    def _step(self, audio: np.ndarray, *, first: bool, last: bool) -> np.ndarray:
        inputs = self.processor(
            audio, sampling_rate=SAMPLE_RATE, is_streaming=True, is_first_audio_chunk=first, is_last_audio_chunk=last
        )
        with TORCH_GPU, self.torch.inference_mode():
            output = self.model(**inputs.to(self.device), speaker_cache=self.cache)
            self.cache = output.speaker_cache
            return output.logits.sigmoid()[0].float().cpu().numpy()

    def _advance(self) -> None:
        self.mel_frame += self.processor.num_mel_frames_per_step
        self.next_start = self.processor.audio_chunk_start(self.mel_frame)
        self.audio = self.audio[self.next_start - self.offset :]
        self.offset = self.next_start

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self.audio = np.concatenate([self.audio, chunk])
        frames = [np.zeros((0, MAX_SPEAKERS), dtype=np.float32)]
        if self.next_start is None:
            if len(self.audio) < self.processor.num_samples_first_audio_chunk:
                return frames[0]
            frames.append(
                self._step(self.audio[: self.processor.num_samples_first_audio_chunk], first=True, last=False)
            )
            self._advance()
        size = self.processor.num_samples_per_audio_chunk
        while len(self.audio) >= size:
            frames.append(self._step(self.audio[:size], first=False, last=False))
            self._advance()
        return np.concatenate(frames)

    def finish(self) -> np.ndarray:
        if self.next_start is None:
            if len(self.audio) == 0:
                return np.zeros((0, MAX_SPEAKERS), dtype=np.float32)
            return self._step(self.audio, first=True, last=True)
        return self._step(self.audio, first=False, last=True)


class WavLMEmbedder:
    """Voice embeddings from Microsoft WavLM x-vectors."""

    def __init__(self, model_id: str, *, device: str) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).to(device).eval()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = audio[: SAMPLE_RATE * 10]
        if len(audio) < MIN_EMBED_SAMPLES:
            audio = np.pad(audio, (0, MIN_EMBED_SAMPLES - len(audio)))
        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with TORCH_GPU, self.torch.inference_mode():
            vector = self.model(**inputs.to(self.device)).embeddings[0].float().cpu().numpy()
        return vector / np.linalg.norm(vector)


class WhisperTranscriber:
    """Local Whisper on Apple silicon through MLX."""

    def __init__(self, model_id: str, *, language: str) -> None:
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_id = model_id
        self.language = language

    def transcribe(self, audio: np.ndarray, hints: Hints) -> str:
        """Transcribes a turn. The terms steer how Whisper spells names and the group's own words, such as
        "EZoo", and the recent lines give a short turn its context. On a short unclear clip, Whisper can repeat
        its prompt instead of hearing the clip, so a turn that only repeats the prompt is heard again with less
        of it: the terms alone, then nothing."""
        terms = ", ".join(hints.terms) + "." if hints.terms else ""
        text = ""
        for prompt in dict.fromkeys([f"{terms} {hints.recent}".strip(), terms, ""]):
            text = self._run(audio, prompt)
            if not echoes(text, prompt):
                return text
        return text

    def _run(self, audio: np.ndarray, prompt: str) -> str:
        result = self.mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_id,
            language=self.language,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        return result["text"].strip()


def echoes(text: str, prompt: str) -> bool:
    """Whether a transcript holds only words of its prompt, such as "Glow, Josh Park, Glow, Josh Park," or the
    last line said once more."""
    said = re.findall(r"[a-z0-9']+", text.lower())
    return len(said) >= 2 and set(said) <= set(re.findall(r"[a-z0-9']+", prompt.lower()))


class OpenAIChat:
    """Any OpenAI-compatible chat server, such as LM Studio."""

    def __init__(self, base_url: str) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key="local", timeout=CHAT_TIMEOUT_SECONDS, max_retries=1)

    async def complete(self, request: dict, on_text: Callable[[str], None] | None = None) -> str:
        """Streams every answer, so the timeout limits each wait for the server and not the length of a long
        answer. A server that hangs fails the call, so it cannot stall the run."""
        text = ""
        async for chunk in await self.client.chat.completions.create(**request, stream=True):
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                text += delta
                if on_text:
                    on_text(text)
        return text


def model_file_hash(path: str | None) -> str:
    """Fingerprints a local model directory by its small files and the names and sizes of its weights."""
    if not path:
        return "unknown"
    digest = hashlib.sha256()
    for file in sorted(Path(path).expanduser().rglob("*")):
        if not file.is_file():
            continue
        digest.update(file.name.encode())
        if file.stat().st_size < 1_000_000:
            digest.update(file.read_bytes())
        else:
            digest.update(json.dumps(file.stat().st_size).encode())
    return digest.hexdigest()[:16]
