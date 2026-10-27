"""Builds a sample conversation MP3 and its hand labels from a script, with macOS `say` voices.

A voice is a `say` voice name, or an object with the name, a pitch shift in semitones, and a rate in words
per minute. Several macOS voices sound alike to speaker models, so a shift keeps the speakers distinct.

A negative `pause_after_ms` starts the next line before this one ends, so the two overlap. A script `noise`
level mixes in seeded room noise and a mains hum at that amplitude, so each build gives the same audio."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from grapevine.audio import SAMPLE_RATE, decode

GAP_MS = 500
SILENCE = 0.003
WINDOW_MS = 10
HUM_HZ = 60
NOISE_SEED = 0


def synthesize(text: str, voice: str | dict, workdir: Path, index: int) -> np.ndarray:
    spec = {"name": voice} if isinstance(voice, str) else voice
    aiff = workdir / f"{index}.aiff"
    rate = ["-r", str(spec["rate"])] if "rate" in spec else []
    subprocess.run(["say", "-v", spec["name"], *rate, "-o", str(aiff), text], check=True)
    audio = decode(aiff)
    if not spec.get("pitch"):
        return audio
    import librosa

    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=spec["pitch"]).astype(np.float32)


def voiced_ms(speech: np.ndarray) -> int:
    """The length of a line up to its last sound. `say` ends each line with silence, which is not speech."""
    window = SAMPLE_RATE * WINDOW_MS // 1000
    loud = [i for i in range(0, len(speech), window) if np.sqrt(np.mean(speech[i : i + window] ** 2)) > SILENCE]
    return (loud[-1] + window) * 1000 // SAMPLE_RATE if loud else 0


def arrange(lines: list[dict], speeches: list[np.ndarray], noise: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    """Mixes the lines onto one track. Each line starts after the previous line's audio plus its pause."""
    labels = []
    starts = []
    cursor_ms = GAP_MS
    for line, speech in zip(lines, speeches, strict=True):
        starts.append(cursor_ms)
        labels.append(
            {
                "speaker": line["speaker"],
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + voiced_ms(speech),
                "text": line["text"],
            }
        )
        cursor_ms = max(GAP_MS, cursor_ms + len(speech) * 1000 // SAMPLE_RATE + line.get("pause_after_ms", GAP_MS))

    end_ms = max([cursor_ms] + [start + len(speech) * 1000 // SAMPLE_RATE for start, speech in zip(starts, speeches)])
    audio = np.zeros(SAMPLE_RATE * end_ms // 1000, dtype=np.float32)
    for start_ms, speech in zip(starts, speeches):
        offset = SAMPLE_RATE * start_ms // 1000
        audio[offset : offset + len(speech)] += speech
    if noise:
        seconds = np.arange(len(audio)) / SAMPLE_RATE
        hiss = np.random.default_rng(NOISE_SEED).normal(0, noise, len(audio))
        audio += (hiss + noise * np.sin(2 * np.pi * HUM_HZ * seconds)).astype(np.float32)
    return np.clip(audio, -1, 1), labels


def build(script_path: Path) -> None:
    script = json.loads(script_path.read_text())
    stem = script_path.name.removesuffix(".script.json")
    with tempfile.TemporaryDirectory() as tmp:
        speeches = [
            synthesize(line["text"], script["voices"][line["speaker"]], Path(tmp), index)
            for index, line in enumerate(script["lines"])
        ]
    audio, labels = arrange(script["lines"], speeches, script.get("noise", 0.0))

    duration_ms = len(audio) * 1000 // SAMPLE_RATE
    mp3 = script_path.with_name(f"{stem}.mp3")
    sf.write(mp3, audio, SAMPLE_RATE, format="MP3")
    intro_end_ms = labels[script["intro_lines"] - 1]["end_ms"] + GAP_MS // 2
    out = {"audio": mp3.name, "intro_end_ms": intro_end_ms, "duration_ms": duration_ms, "turns": labels}
    script_path.with_name(f"{stem}.labels.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {mp3} ({duration_ms / 1000:.1f}s, {len(labels)} turns)")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
