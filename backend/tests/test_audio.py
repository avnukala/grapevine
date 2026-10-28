import asyncio
from pathlib import Path

import numpy as np

from grapevine import audio


def on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_a_replay_decodes_its_file_off_the_event_loop(monkeypatch):
    decoded_on_loop = []

    def decode(path: Path) -> np.ndarray:
        decoded_on_loop.append(on_event_loop())
        return np.zeros(audio.CHUNK_SAMPLES * 2, dtype=np.float32)

    monkeypatch.setattr(audio, "decode", decode)

    async def replay() -> list[np.ndarray]:
        return [chunk async for chunk in audio.file_chunks(Path("talk.wav"), speed=None, stop=asyncio.Event())]

    assert len(asyncio.run(replay())) == 2
    assert decoded_on_loop == [False]
