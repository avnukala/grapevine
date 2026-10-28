"""Audio input. A live microphone and a replayed file both yield the same 16 kHz mono chunks."""

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


def decode(path: Path) -> np.ndarray:
    """Decodes a WAV, AIFF, FLAC, OGG, or MP3 file to 16 kHz mono float32."""
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    return soxr.resample(audio.mean(axis=1), rate, SAMPLE_RATE).astype(np.float32)


async def file_chunks(path: Path, *, speed: float | None, stop: asyncio.Event) -> AsyncIterator[np.ndarray]:
    """Yields chunks of a file. With a speed, the file streams in real time scaled by it, else as fast as possible."""
    audio = await asyncio.to_thread(decode, path)
    loop = asyncio.get_running_loop()
    started = loop.time()
    for offset in range(0, len(audio), CHUNK_SAMPLES):
        if stop.is_set():
            return
        if speed:
            due = started + offset / SAMPLE_RATE / speed
            await asyncio.sleep(max(0.0, due - loop.time()))
        else:
            await asyncio.sleep(0)
        yield audio[offset : offset + CHUNK_SAMPLES]


async def mic_chunks(record_path: Path, *, stop: asyncio.Event) -> AsyncIterator[np.ndarray]:
    """Yields microphone chunks and writes them to `record_path`, so the live session can be replayed."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def on_audio(data: np.ndarray, frames: int, time: object, status: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, data[:, 0].copy())

    record_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        sf.SoundFile(record_path, mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as recording,
        sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK_SAMPLES, callback=on_audio),
    ):
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            recording.write(chunk)
            yield chunk
