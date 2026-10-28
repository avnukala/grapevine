import numpy as np

from grapevine.models import Hints, WhisperTranscriber, echoes


def test_a_transcript_of_only_prompt_words_is_an_echo():
    prompt = "Mia Rao, Glow, EZoo. We went to Glow in 2024."

    assert echoes("Glow, Mia Rao, Glow, Mia Rao,", prompt)
    assert echoes("Glow in 2024.", prompt)
    assert not echoes("Close to 24.", prompt)
    assert not echoes("EZoo.", prompt)


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
