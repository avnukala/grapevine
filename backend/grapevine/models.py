"""The audio and language models behind small interfaces, so a test can swap each one for a fake."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from grapevine.audio import SAMPLE_RATE

FRAME_MS = 10
MAX_SPEAKERS = 8
MIN_EMBED_SAMPLES = SAMPLE_RATE // 2


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


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class ChatModel(Protocol):
    async def complete(self, request: dict) -> str:
        """Sends an OpenAI chat completion request and returns the message content."""


@dataclass
class Models:
    detector: SpeakerDetector
    embedder: VoiceEmbedder
    transcriber: Transcriber
    chat: ChatModel


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
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs, speaker_cache=self.cache)
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

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = WavLMForXVector.from_pretrained(model_id).eval()

    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = audio[: SAMPLE_RATE * 10]
        if len(audio) < MIN_EMBED_SAMPLES:
            audio = np.pad(audio, (0, MIN_EMBED_SAMPLES - len(audio)))
        inputs = self.extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with self.torch.inference_mode():
            vector = self.model(**inputs).embeddings[0].numpy()
        return vector / np.linalg.norm(vector)


class WhisperTranscriber:
    """Local Whisper on Apple silicon through MLX."""

    def __init__(self, model_id: str, *, language: str) -> None:
        import mlx_whisper

        self.mlx_whisper = mlx_whisper
        self.model_id = model_id
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        result = self.mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_id,
            language=self.language,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return result["text"].strip()


class OpenAIChat:
    """Any OpenAI-compatible chat server, such as LM Studio."""

    def __init__(self, base_url: str) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key="local")

    async def complete(self, request: dict) -> str:
        response = await self.client.chat.completions.create(**request)
        return response.choices[0].message.content or ""


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
