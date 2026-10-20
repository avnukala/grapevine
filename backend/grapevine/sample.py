"""Builds a sample conversation MP3 and its hand labels from a script, with macOS `say` voices."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from grapevine.audio import SAMPLE_RATE, decode

GAP_MS = 500


def synthesize(text: str, voice: str, workdir: Path, index: int) -> np.ndarray:
    aiff = workdir / f"{index}.aiff"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    return decode(aiff)


def build(script_path: Path) -> None:
    script = json.loads(script_path.read_text())
    stem = script_path.name.removesuffix(".script.json")
    gap = np.zeros(SAMPLE_RATE * GAP_MS // 1000, dtype=np.float32)
    parts = [gap]
    labels = []
    cursor_ms = GAP_MS

    with tempfile.TemporaryDirectory() as tmp:
        for index, line in enumerate(script["lines"]):
            speech = synthesize(line["text"], script["voices"][line["speaker"]], Path(tmp), index)
            duration_ms = len(speech) * 1000 // SAMPLE_RATE
            labels.append(
                {
                    "speaker": line["speaker"],
                    "start_ms": cursor_ms,
                    "end_ms": cursor_ms + duration_ms,
                    "text": line["text"],
                }
            )
            pause_ms = line.get("pause_after_ms", GAP_MS)
            parts += [speech, np.zeros(SAMPLE_RATE * pause_ms // 1000, dtype=np.float32)]
            cursor_ms += duration_ms + pause_ms

    audio = np.concatenate(parts)
    mp3 = script_path.with_name(f"{stem}.mp3")
    sf.write(mp3, audio, SAMPLE_RATE, format="MP3")
    intro_end_ms = labels[script["intro_lines"] - 1]["end_ms"] + GAP_MS // 2
    out = {"audio": mp3.name, "intro_end_ms": intro_end_ms, "duration_ms": cursor_ms, "turns": labels}
    script_path.with_name(f"{stem}.labels.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {mp3} ({cursor_ms / 1000:.1f}s, {len(labels)} turns)")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
