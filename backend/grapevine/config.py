"""Settings for the process and the fixed configuration of each run."""

import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

from grapevine.turns import TurnRules

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("GRAPEVINE_DATA", ROOT / "data"))
DB_PATH = Path(os.environ.get("GRAPEVINE_DB", DATA_DIR / "grapevine.db"))
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH")

Context = Literal["full", "relevant"]


@dataclass(frozen=True)
class RunConfig:
    """Everything that can change the output of a run. A replay with a new value is a new run."""

    diarizer: str = "nvidia/Nemotron-3-Diarization"
    diarizer_mode: str = "low_latency"
    diarizer_device: str = "mps"
    embedder: str = "microsoft/wavlm-base-plus-sv"
    voice_threshold: float = 0.86
    new_voice_threshold: float = 0.93
    transcriber: str = "mlx-community/whisper-large-v3-turbo"
    language: str = "en"
    turn_rules: TurnRules = field(default_factory=TurnRules)
    extract_model: str = os.environ.get("LLM_MODEL", "gemma-4-26b-a4b")
    summary_model: str = os.environ.get("SUMMARY_MODEL", os.environ.get("LLM_MODEL", "gemma-4-26b-a4b"))
    seed: int = 7
    context: Context = "full"
    prompt_pause_ms: int = 7000
    reuse_model_calls: bool = True

    def to_json(self) -> dict:
        return asdict(self)


def run_config(overrides: dict) -> RunConfig:
    rules = TurnRules(**overrides.pop("turn_rules", {}))
    return replace(RunConfig(), turn_rules=rules, **overrides)


def code_version() -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}-dirty" if git("status", "--porcelain", "--untracked-files=no") else sha
