import numpy as np

from grapevine.audio import SAMPLE_RATE
from grapevine.sample import GAP_MS, arrange


def tone(ms: int) -> np.ndarray:
    return np.full(SAMPLE_RATE * ms // 1000, 0.1, dtype=np.float32)


def test_lines_follow_each_other_after_a_pause():
    lines = [{"speaker": "Marcus Hale", "text": "Hi."}, {"speaker": "Tina Alvarez", "text": "Hey."}]
    audio, labels = arrange(lines, [tone(1000), tone(500)])
    assert [(t["start_ms"], t["end_ms"]) for t in labels] == [(500, 1500), (2000, 2500)]
    assert len(audio) == SAMPLE_RATE * (2500 + GAP_MS) // 1000


def test_negative_pause_overlaps_the_next_line_and_keeps_both_labels():
    lines = [
        {"speaker": "Marcus Hale", "text": "So this is my brother", "pause_after_ms": -400},
        {"speaker": "Tina Alvarez", "text": "Yeah", "pause_after_ms": 0},
    ]
    audio, labels = arrange(lines, [tone(1000), tone(200)])
    assert [(t["speaker"], t["start_ms"], t["end_ms"]) for t in labels] == [
        ("Marcus Hale", 500, 1500),
        ("Tina Alvarez", 1100, 1300),
    ]
    assert audio[SAMPLE_RATE * 1200 // 1000] == np.float32(0.2)
    assert len(audio) == SAMPLE_RATE * 1500 // 1000


def test_noise_is_seeded_and_fills_the_silence():
    lines = [{"speaker": "Dana Cole", "text": "Mm-hm."}]
    first, _ = arrange(lines, [tone(300)], noise=0.01)
    second, _ = arrange(lines, [tone(300)], noise=0.01)
    assert np.array_equal(first, second)
    assert 0.005 < np.sqrt(np.mean(first[: SAMPLE_RATE * GAP_MS // 1000] ** 2)) < 0.03
