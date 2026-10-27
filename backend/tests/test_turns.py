import numpy as np

from grapevine.models import MAX_SPEAKERS
from grapevine.turns import Turn, TurnBuilder, TurnRules


def frames(*spans: tuple[int | None, int]) -> np.ndarray:
    """Builds frames from (speaker or None for silence, duration in ms) spans."""
    rows = []
    for speaker, ms in spans:
        for _ in range(ms // 10):
            row = np.zeros(MAX_SPEAKERS, dtype=np.float32)
            if speaker is not None:
                row[speaker] = 0.9
            rows.append(row)
    return np.array(rows)


def test_turn_ends_after_silence():
    builder = TurnBuilder(TurnRules(end_silence_ms=500))
    turns = builder.push(frames((None, 200), (0, 1000), (None, 300), (0, 500), (None, 600)))
    assert turns == [Turn(0, 200, 2000, pauses=(1350,))]


def test_new_speaker_ends_the_turn_once_they_hold_the_floor():
    builder = TurnBuilder(TurnRules(switch_ms=200))
    turns = builder.push(frames((0, 1000), (1, 100), (0, 500), (1, 1000)))
    turns += builder.close()
    assert turns == [Turn(0, 0, 1600), Turn(1, 1600, 2600)]


def test_short_blips_are_not_turns():
    builder = TurnBuilder(TurnRules(min_turn_ms=400))
    assert builder.push(frames((0, 300), (None, 1000))) == []


def test_long_turn_splits_at_a_pause():
    builder = TurnBuilder(TurnRules(max_turn_ms=3000, split_pause_ms=250))
    turns = builder.push(frames((0, 2000), (None, 300), (0, 2000), (None, 1000)))
    assert turns == [Turn(0, 0, 2150), Turn(0, 2150, 4300)]


def test_a_voice_the_detector_splits_across_labels_is_still_speech():
    unsure = np.zeros((60, MAX_SPEAKERS), dtype=np.float32)
    unsure[:, 2], unsure[:, 4] = 0.45, 0.2
    builder = TurnBuilder(TurnRules())
    turns = builder.push(np.concatenate([frames((2, 500)), unsure, frames((None, 1000))]))
    assert turns == [Turn(2, 0, 1100)]


def test_silence_counts_from_the_last_speech():
    builder = TurnBuilder(TurnRules())
    builder.push(frames((0, 1000), (None, 7000)))
    assert builder.silence_ms == 7000
