import asyncio
import json
import re

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache
from openai import APITimeoutError

from grapevine import models
from grapevine.audio import SAMPLE_RATE
from grapevine.config import RunConfig
from grapevine.models import Hints, OpenAIChat, WavLMEmbedder, WhisperTranscriber, echoes


def test_a_transcript_of_only_prompt_words_is_an_echo():
    prompt = "Mia Rao, Glow, EZoo. We went to Glow in 2024."

    assert echoes("Glow, Mia Rao, Glow, Mia Rao,", prompt)
    assert echoes("Glow in 2024.", prompt)
    assert not echoes("Close to 24.", prompt)
    assert not echoes("EZoo.", prompt)


def test_wavlm_gives_each_clip_of_a_call_the_voice_it_gets_alone():
    model_id = RunConfig().embedder
    if not isinstance(try_to_load_from_cache(model_id, "config.json"), str):
        pytest.skip("WavLM is not downloaded")
    embedder = WavLMEmbedder(model_id, device="cpu")
    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 3).astype(np.float32) * 0.1
    clips = [noise[: SAMPLE_RATE // 4], noise, noise[:SAMPLE_RATE]]

    together = embedder.embed(clips)

    np.testing.assert_allclose(together, [embedder.embed([clip])[0] for clip in clips], atol=1e-5)


def test_an_echo_is_heard_again_with_less_of_the_prompt():
    prompts = []

    class FakeWhisper:
        def transcribe(self, audio, *, initial_prompt, **options):
            prompts.append(initial_prompt)
            echo = {"Mia Rao, Glow. We went in 2023.": "We went in 2023.", "Mia Rao, Glow.": "Glow, Mia Rao, Glow,"}
            return {"text": echo.get(initial_prompt, "Close to 24.")}

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.mlx_whisper, transcriber.model_id, transcriber.language = FakeWhisper(), "test", "en"

    text = transcriber.transcribe(np.zeros(1600), Hints(["Mia Rao", "Glow"], "We went in 2023."))

    assert text == "Close to 24."
    assert prompts == ["Mia Rao, Glow. We went in 2023.", "Mia Rao, Glow.", None]


async def serve(pieces: list[str], *, gap_seconds: float):
    """A chat server that streams each piece after a gap, then ends the answer."""

    async def answer(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        size = int(re.search(rb"content-length: (\d+)", head.lower())[1])
        await reader.readexactly(size)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
        for piece in pieces:
            await asyncio.sleep(gap_seconds)
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m"}
            chunk["choices"] = [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]
            writer.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await writer.drain()
        writer.write(b"data: [DONE]\n\n")
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    return server, f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"


REQUEST = {"model": "m", "messages": [{"role": "user", "content": "Hi"}]}


def test_a_long_answer_that_keeps_coming_outlasts_the_timeout(monkeypatch):
    monkeypatch.setattr(models, "CHAT_TIMEOUT_SECONDS", 0.3)

    async def ask():
        server, url = await serve(["Marcus", " Hale", " met", " Dana", " Cole"], gap_seconds=0.1)
        try:
            return await asyncio.wait_for(OpenAIChat(url).complete(REQUEST), 5)
        finally:
            server.close()

    assert asyncio.run(ask()) == "Marcus Hale met Dana Cole"


def test_a_server_that_stops_answering_fails_the_call(monkeypatch):
    monkeypatch.setattr(models, "CHAT_TIMEOUT_SECONDS", 0.3)

    async def ask():
        server, url = await serve(["Marcus"], gap_seconds=60)
        try:
            return await asyncio.wait_for(OpenAIChat(url).complete(REQUEST), 5)
        finally:
            server.close()

    with pytest.raises(APITimeoutError):
        asyncio.run(ask())
