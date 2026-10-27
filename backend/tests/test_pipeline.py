import asyncio
from dataclasses import replace

import numpy as np
import pytest

from grapevine import pipeline as pipeline_module
from grapevine.audio import SAMPLE_RATE
from grapevine.config import RunConfig
from grapevine.llm import LLM
from grapevine.pipeline import Pipeline, RunStopping, said_name
from grapevine.store import Store
from tests.fakes import FakeChat, Line, chunks, fact, fake_models, render

LINES = [
    Line(0, 500, 2000, "Hey, I'm Ray Patel."),
    Line(1, 2500, 4000, "I'm Jasper Lin."),
    Line(2, 4500, 6000, "Hi, I'm Sophie Moreau."),
    Line(1, 7000, 9000, "Ray and I met at Cornell in 2016."),
    Line(3, 9500, 11000, "Hello there."),
    Line(0, 20000, 22000, "We climb on Tuesdays."),
]
INTRO_END_MS = 6500

RULES = {
    "Ray and I met": {
        "facts": [
            fact(
                "Ray Patel and Jasper Lin met at Cornell in 2016.",
                ["Ray Patel", "speaker"],
                kind="relationship",
                start_date="2016",
            )
        ],
        "relationships": [
            {
                "a": "speaker",
                "b": "Ray Patel",
                "category": "friend",
                "label": "college friend",
                "status": "current",
                "met_on": None,
                "met_event": None,
                "closeness": 4,
                "recency": 5,
                "sentiment": 1,
                "summary": "Met at Cornell.",
                "facts": [0],
            }
        ],
    },
    "I'm Ray Patel": {"speaker_name": "Ray Patel"},
    "I'm Jasper Lin": {"speaker_name": "Jasper Lin"},
    "I'm Sophie Moreau": {"speaker_name": "Sophie Moreau"},
}
PROMPTS = [{"text": "Sophie and Ray, how did you meet?", "reason": "pause", "people": ["p1", "p3"]}]


def replay(conn, chat, merged=None, *, lines=LINES, roster=()):
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
    ).lastrowid
    store = Store(conn, run_id)
    llm = LLM(conn, chat, model_file_hash="test", reuse=True)
    pipeline = Pipeline(
        store,
        fake_models(lines, chat, merged),
        llm,
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=INTRO_END_MS,
        roster=list(roster),
    )
    asyncio.run(pipeline.run(chunks(render(lines, 24000))))
    return store


def snapshot(store):
    graph = store.graph
    name = {pid: p["name"] for pid, p in graph.people.items()}
    return {
        "segments": [
            (s["text"], name.get(s["person_id"])) for s in sorted(graph.segments.values(), key=lambda s: s["id"])
        ],
        "facts": sorted(f["text"] for f in graph.facts.values()),
        "relationships": sorted((name[r["a"]], name[r["b"]], r["category"]) for r in graph.relationships.values()),
        "prompts": sorted((p["reason"], p["text"]) for p in graph.prompts.values()),
        "phase": graph.phase,
    }


def test_replay_names_voices_flags_unknown_speakers_and_extracts(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    store = replay(conn, FakeChat(RULES, PROMPTS))

    result = snapshot(store)
    assert result["segments"] == [
        ("Hey, I'm Ray Patel.", "Ray Patel"),
        ("I'm Jasper Lin.", "Jasper Lin"),
        ("Hi, I'm Sophie Moreau.", "Sophie Moreau"),
        ("Ray and I met at Cornell in 2016.", "Jasper Lin"),
        ("Hello there.", None),
        ("We climb on Tuesdays.", "Ray Patel"),
    ]
    assert result["relationships"] == [("Ray Patel", "Jasper Lin", "friend")]
    assert [s["speaker_label"] for s in store.graph.segments.values()][:3] == ["spk0", "spk1", "spk2"]
    assert result["prompts"] == [("pause", "Sophie and Ray, how did you meet?")]
    assert result["phase"] == "ended"
    unknown = next(s for s in store.graph.segments.values() if s["text"] == "Hello there.")
    assert unknown["speaker_confidence"] < RunConfig().voice_threshold


def test_unchanged_replay_reuses_every_model_call(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    chat = FakeChat(RULES, PROMPTS)
    first = replay(conn, chat)
    calls = len(chat.requests)
    second = replay(conn, chat)

    assert len(chat.requests) == calls
    cached = conn.execute("SELECT MIN(cached) FROM model_calls WHERE run_id = ?", (second.run_id,)).fetchone()[0]
    assert cached == 1
    assert snapshot(second) == snapshot(first)


def test_voices_split_people_the_speaker_detector_merged(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    store = replay(conn, FakeChat(RULES, PROMPTS), merged={2: 0})

    segments = snapshot(store)["segments"]
    assert segments[2] == ("Hi, I'm Sophie Moreau.", "Sophie Moreau")
    assert segments[5] == ("We climb on Tuesdays.", "Ray Patel")
    assert len({pid for pid in store.graph.speakers.values() if pid is not None}) == 3


def test_a_turn_that_two_merged_voices_share_splits_by_voice(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [
        Line(0, 500, 2700, "Hey, I'm Ray Patel."),
        Line(2, 3000, 5200, "Hi, I'm Sophie Moreau."),
        Line(1, 5700, 7200, "I'm Jasper Lin."),
        Line(2, 8000, 10000, "Ray and I met at Cornell in 2016."),
    ]
    chat = FakeChat({**RULES, "I'm Ray Patel": {"speaker_name": "Ray Patel"}})
    store = Store(
        conn,
        conn.execute(
            "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
        ).lastrowid,
    )
    pipeline = Pipeline(
        store,
        fake_models(lines, chat, {2: 0}),
        LLM(conn, chat, model_file_hash="t", reuse=True),
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=7600,
        roster=[],
    )
    asyncio.run(pipeline.run(chunks(render(lines, 11000))))

    assert snapshot(store)["segments"] == [
        ("Hey, I'm Ray Patel.", "Ray Patel"),
        ("Hi, I'm Sophie Moreau.", "Sophie Moreau"),
        ("I'm Jasper Lin.", "Jasper Lin"),
        ("Ray and I met at Cornell in 2016.", "Sophie Moreau"),
    ]


def test_a_new_name_under_a_named_voice_gets_its_own_voice(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [
        Line(0, 500, 2700, "Hey, I'm Ray Patel."),
        Line(2, 3700, 5200, "Hi, I'm Sophie Moreau."),
        Line(1, 5700, 7200, "I'm Jasper Lin."),
        Line(2, 8000, 10000, "Ray and I met at Cornell in 2016."),
    ]
    chat = FakeChat(RULES)
    store = Store(
        conn,
        conn.execute(
            "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
        ).lastrowid,
    )
    pipeline = Pipeline(
        store,
        fake_models(lines, chat, {2: 0}),
        LLM(conn, chat, model_file_hash="t", reuse=True),
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=7600,
        roster=[],
    )
    asyncio.run(pipeline.run(chunks(render(lines, 11000))))

    assert [name for _, name in snapshot(store)["segments"]] == [
        "Ray Patel",
        "Sophie Moreau",
        "Jasper Lin",
        "Sophie Moreau",
    ]


def test_a_self_introduction_respells_a_known_person(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    conn.execute("INSERT INTO people (name, aliases) VALUES ('Sophiely Moreau', '[\"Sophie\", \"Sophie Moreau\"]')")
    store = replay(conn, FakeChat(RULES))

    sophie = store.person(1)
    assert snapshot(store)["segments"][2] == ("Hi, I'm Sophie Moreau.", "Sophie Moreau")
    assert sophie["name"] == "Sophie Moreau"
    assert "Sophiely Moreau" in sophie["aliases"]


def test_the_roster_seats_people_and_fixes_how_their_names_are_spelled(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [replace(LINES[0], text="Hey, I'm Ray Patell."), replace(LINES[1], text="I'm Jasper Lind."), *LINES[2:]]
    store = replay(conn, FakeChat(RULES), lines=lines, roster=["Ray Patel", "Jasper Lin", "Sophie Moreau"])

    names = {pid: p["name"] for pid, p in store.graph.people.items()}
    assert [names[pid] for pid in store.graph.roster] == ["Ray Patel", "Jasper Lin", "Sophie Moreau"]
    assert snapshot(store)["segments"][:3] == [
        ("Hey, I'm Ray Patel.", "Ray Patel"),
        ("I'm Jasper Lin.", "Jasper Lin"),
        ("Hi, I'm Sophie Moreau.", "Sophie Moreau"),
    ]
    assert sorted(names.values()) == ["Jasper Lin", "Ray Patel", "Sophie Moreau"]


def test_a_short_introduction_that_sounds_unlike_its_label_gets_its_own_voice(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [
        Line(0, 500, 2700, "Hey, I'm Ray Patel."),
        Line(1, 3200, 5600, "I love climbing."),
        Line(2, 6600, 8000, "I'm Jasper Lin."),
    ]
    chat = FakeChat(RULES)
    store = Store(
        conn,
        conn.execute(
            "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
        ).lastrowid,
    )
    pipeline = Pipeline(
        store,
        fake_models(lines, chat, {2: 1}),
        LLM(conn, chat, model_file_hash="t", reuse=True),
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=8500,
        roster=[],
    )
    asyncio.run(pipeline.run(chunks(render(lines, 9500))))

    assert [name for _, name in snapshot(store)["segments"]] == ["Ray Patel", None, "Jasper Lin"]


class SecondsTranscriber:
    def transcribe(self, audio, names):
        return f"{len(audio) // SAMPLE_RATE} seconds"


async def paced(audio):
    async for chunk in chunks(audio):
        await asyncio.sleep(0.02)
        yield chunk


def test_a_run_at_real_time_previews_the_open_turn(conn, monkeypatch):
    monkeypatch.setattr(pipeline_module, "PARTIAL_MS", 200)
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [Line(0, 500, 4500, "")]
    messages = []
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'live', '{}', 'test')"
    ).lastrowid
    chat = FakeChat({})
    models = replace(fake_models(lines, chat), transcriber=SecondsTranscriber())
    llm = LLM(conn, chat, model_file_hash="t", reuse=True)
    pipeline = Pipeline(
        Store(conn, run_id, messages.append), models, llm, RunConfig(), today="2025-10-18", intro_end_ms=0, roster=[]
    )
    pipeline.partials = True
    asyncio.run(pipeline.run(paced(render(lines, 6000))))

    previews = [m for m in messages if m["type"] == "partial"]
    assert "spk0" in {m["speaker_label"] for m in previews if m["text"]}
    assert len({m["text"] for m in previews if m["text"]}) > 1
    assert previews[-1] == {"type": "partial", "start_ms": 500, "text": None, "run_id": run_id}


def test_a_leading_full_name_names_a_voice_only_when_it_is_on_the_roster():
    def heard(text, roster):
        return said_name(text, None, intro=True, unnamed=True, roster=roster)

    assert heard("Chris Porter, I'm an accountant.", []) == "Chris Porter"
    assert heard("Most Sundays.", []) == "Most Sundays"
    assert heard("Most Sundays.", ["Chris Porter"]) is None
    assert heard("Chris Porter, I'm an accountant.", ["Chris Porter"]) == "Chris Porter"


def test_the_review_completes_relationships_groups_and_events(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")

    def rel(a, b, category, met_on=None):
        return {
            "a": a,
            "b": b,
            "category": category,
            "label": category,
            "status": "current",
            "met_on": met_on,
            "in_touch": True,
            "summary": "",
        }

    reviewed = {
        "room": {
            "Ray Patel and Jasper Lin": [rel("Ray Patel", "Jasper Lin", "professional", "2016")],
            "Ray Patel and Sophie Moreau": [rel("Ray Patel", "Sophie Moreau", "friend")],
        },
        "other": [],
        "known": {},
        "candidates": {},
        "new": [{"name": "Tuesday crew", "also_called": ["climb"], "members": ["Ray Patel", "Jasper Lin"]}],
        "happenings": [
            {"name": "Electric Zoo 2018", "record": None, "start_date": "2018", "place": None, "people": ["Ray Patel"]}
        ],
        "members": ["Sophie Moreau"],
    }
    store = replay(conn, FakeChat(RULES, reviewed=reviewed))
    graph = store.graph
    names = {pid: p["name"] for pid, p in graph.people.items()}

    assert sorted((names[r["a"]], names[r["b"]], r["category"], r["met_on"]) for r in graph.relationships.values()) == [
        ("Ray Patel", "Jasper Lin", "professional", "2016"),
        ("Ray Patel", "Sophie Moreau", "friend", None),
    ]
    [group] = graph.groups.values()
    assert {names[m["person_id"]] for m in graph.members.values() if m["group_id"] == group["id"]} == {
        "Ray Patel",
        "Jasper Lin",
        "Sophie Moreau",
    }
    assert [e["name"] for e in graph.events.values()] == ["Electric Zoo 2018"]


class DownChat(FakeChat):
    async def complete(self, request: dict, on_text=None) -> str:
        raise ConnectionError("no model loaded")


def test_a_typed_run_fails_on_its_first_turn_when_the_model_is_down(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'live', '{}', 'test')"
    ).lastrowid
    chat = DownChat({})
    llm = LLM(conn, chat, model_file_hash="t", reuse=True)
    pipeline = Pipeline(
        Store(conn, run_id), fake_models([], chat), llm, RunConfig(), today="2025-10-18", intro_end_ms=0, roster=[]
    )
    stop = asyncio.Event()

    async def idle():
        await stop.wait()
        return
        yield

    async def go():
        run = asyncio.create_task(pipeline.run(idle()))
        await pipeline.add_text("Ray Patel", "I work as a nurse in Denver.")
        await asyncio.wait_for(run, timeout=5)

    with pytest.raises(ConnectionError):
        asyncio.run(go())


def test_a_typed_turn_after_the_run_stops_is_refused(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'live', '{}', 'test')"
    ).lastrowid
    chat = FakeChat({})
    llm = LLM(conn, chat, model_file_hash="t", reuse=True)
    pipeline = Pipeline(
        Store(conn, run_id), fake_models([], chat), llm, RunConfig(), today="2025-10-18", intro_end_ms=0, roster=[]
    )

    async def no_chunks():
        return
        yield

    asyncio.run(pipeline.run(no_chunks()))

    with pytest.raises(RunStopping):
        asyncio.run(pipeline.add_text("Marcus Hale", "One more thing."))
    assert pipeline.store.graph.segments == {}


def test_an_introduction_read_under_another_speaker_is_read_again(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    lines = [
        Line(0, 500, 2700, "Hey, I'm Ray Patel."),
        Line(2, 3700, 5200, "Hi, I'm Sophie Moreau."),
        Line(1, 5700, 7200, "I'm Jasper Lin."),
    ]
    wrong = fact("The speaker's name is Sophie Moreau, not Ray Patel.", ["speaker"])
    chat = FakeChat(
        {"speaker: Ray Patel):\nHi, I'm Sophie": {"speaker_name": "Sophie Moreau", "facts": [wrong]}, **RULES}
    )
    store = Store(
        conn,
        conn.execute(
            "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
        ).lastrowid,
    )
    pipeline = Pipeline(
        store,
        fake_models(lines, chat, {2: 0}),
        LLM(conn, chat, model_file_hash="t", reuse=True),
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=7600,
        roster=[],
    )
    asyncio.run(pipeline.run(chunks(render(lines, 9000))))

    turns = [r["messages"][1]["content"].split("NEW TURN", 1)[-1] for r in chat.requests]
    reads = [
        next(i for i, turn in enumerate(turns) if key in turn) for key in ("Ray Patel):\nHi", "Moreau):\nHi", "Jasper")
    ]
    assert reads == sorted(reads)
    assert [name for _, name in snapshot(store)["segments"]] == ["Ray Patel", "Sophie Moreau", "Jasper Lin"]
    assert snapshot(store)["facts"] == []


def test_settling_moves_a_turn_only_to_a_clearly_closer_voice(conn):
    conn.execute("INSERT INTO sessions (name, recorded_at) VALUES ('test', '2025-10-18')")
    store = Store(
        conn,
        conn.execute(
            "INSERT INTO runs (session_id, kind, config_json, code_version) VALUES (1, 'replay', '{}', 'test')"
        ).lastrowid,
    )
    chat = FakeChat({})
    pipeline = Pipeline(
        store,
        fake_models([], chat),
        LLM(conn, chat, model_file_hash="t", reuse=True),
        RunConfig(),
        today="2025-10-18",
        intro_end_ms=0,
        roster=[],
    )
    marcus, felix = (
        store.create_person(name=n, aliases=[], gender="male")["id"] for n in ("Marcus Hale", "Felix Ward")
    )

    def turn(person_id, voice, ms):
        voice = np.array(voice, dtype=np.float32)
        segment = store.add_segment(
            start_ms=0, end_ms=ms, speaker_label="spk0", person_id=person_id, confidence=0.9, text="Hi.", timings={}
        )
        pipeline.segment_voices[segment["id"]] = (voice / np.linalg.norm(voice), ms)
        return segment["id"]

    turn(marcus, [1, 0], 5000)
    turn(felix, [0.8, 0.6], 5000)
    near_tie = turn(marcus, [0.94, 0.33], 800)
    clear = turn(marcus, [0.7, 0.7], 800)
    pipeline._settle_speakers()

    assert store.graph.segments[near_tie]["person_id"] == marcus
    assert store.graph.segments[clear]["person_id"] == felix
