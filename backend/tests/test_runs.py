from grapevine import runs
from grapevine.config import RunConfig


def test_loading_models_fingerprints_the_llm_files(monkeypatch):
    fingerprints = []
    monkeypatch.setattr(runs, "llm_file_hash", lambda: fingerprints.append("t"))
    for loader in ["detector", "embedder", "transcriber", "chat"]:
        monkeypatch.setattr(runs, loader, lambda *args: None)

    runs.load_models(RunConfig())

    assert fingerprints == ["t"]
