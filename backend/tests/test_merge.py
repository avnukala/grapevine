import asyncio

from grapevine.extract import Held, Teller, apply_output
from grapevine.llm import LLM
from grapevine.merge import merge, resolve_merge
from grapevine.store import Store
from tests.fakes import FakeChat, empty_output, fact


def accepted_run(conn, session_name):
    session_id = conn.execute(
        "INSERT INTO sessions (name, recorded_at) VALUES (?, '2025-10-18')", (session_name,)
    ).lastrowid
    run_id = conn.execute(
        "INSERT INTO runs (session_id, kind, config_json, code_version, accepted) VALUES (?, 'replay', '{}', 't', 1)",
        (session_id,),
    ).lastrowid
    return Store(conn, run_id)


def tell(store, speaker_id, text, **output):
    segment = store.add_segment(
        start_ms=0, end_ms=1000, speaker_label="spk0", person_id=speaker_id, confidence=None, text=text, timings={}
    )
    apply_output(store, Held({**empty_output(), **output}, Teller(segment["id"], speaker_id, 1.0, None)))


def friends(a, b, met_on):
    return {
        "a": a,
        "b": b,
        "category": "friend",
        "label": "friend",
        "status": "current",
        "met_on": met_on,
        "met_event": None,
        "closeness": None,
        "recency": None,
        "sentiment": None,
        "summary": "",
        "facts": [0],
    }


def merge_run(conn, chat):
    store = Store(
        conn, conn.execute("INSERT INTO runs (kind, config_json, code_version) VALUES ('merge', '{}', 't')").lastrowid
    )
    asyncio.run(merge(store, LLM(conn, chat, model_file_hash="t", reuse=False), model="m", seed=1))
    return store


def test_merge_combines_sessions_and_sends_only_conflicts_to_review(conn):
    first = accepted_run(conn, "one")
    ray = first.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    tell(
        first,
        ray,
        "Jasper and I met in 2016.",
        facts=[fact("Ray Patel and Jasper Lin met in 2016.", ["speaker", "Jasper Lin"], start_date="2016")],
        relationships=[friends("speaker", "Jasper Lin", "2016")],
    )
    second = accepted_run(conn, "two")
    tell(
        second,
        ray,
        "Jasper and I met in 2015.",
        facts=[fact("Ray Patel and Jasper Lin met in 2015.", ["speaker", "Jasper Lin"], start_date="2015")],
        relationships=[friends("speaker", "Jasper", "2015")],
    )

    store = merge_run(conn, FakeChat({}))

    [rel] = store.graph.relationships.values()
    assert rel["met_on"] == "2015" and len(rel["evidence"]) == 2
    [conflict] = store.graph.active_prompts()
    assert conflict["reason"] == "merge_conflict"

    resolve_merge(store, conflict["id"], action="remove_edge", fact_id=None)
    assert store.graph.relationships == {}
    assert store.graph.active_prompts() == []


def two_andrews(conn):
    run = accepted_run(conn, "one")
    ray = run.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    tell(run, ray, "Andrew Gao works with Jasper.", facts=[fact("Andrew Gao works at Mount Sinai.", ["Andrew Gao"])])
    andrew = run.create_person(name="Andrew", aliases=[], gender="male")["id"]
    gao = next(p for p in run.graph.people.values() if p["name"] == "Andrew Gao")["id"]
    return gao, andrew


def same_person(confidence):
    return FakeChat(
        {},
        merged={
            "relationships": [],
            "conflicts": [],
            "same_person": [{"a": "p2", "b": "p3", "confidence": confidence}],
        },
    )


def test_merge_joins_people_it_is_sure_about(conn):
    gao, andrew = two_andrews(conn)
    merge_run(conn, same_person(0.95))
    assert conn.execute("SELECT merged_into_id FROM people WHERE id = ?", (andrew,)).fetchone()[0] == gao


def test_merge_asks_about_people_it_is_not_sure_about(conn):
    gao, andrew = two_andrews(conn)
    store = merge_run(conn, same_person(0.5))
    [prompt] = store.graph.active_prompts()
    assert prompt["text"] == "Is Andrew the same person as Andrew Gao?"

    resolve_merge(store, prompt["id"], action="merge_people", fact_id=None)

    assert conn.execute("SELECT merged_into_id FROM people WHERE id = ?", (andrew,)).fetchone()[0] == gao


def test_misspelled_names_are_candidates_for_the_same_person(conn):
    one, two = accepted_run(conn, "one"), accepted_run(conn, "two")
    ray = one.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    tell(one, ray, "Sam Torres climbs.", facts=[fact("Sam Torres climbs.", ["Sam Torres"])])
    two.show_person(ray)
    two.create_person(name="Sam Taurus", aliases=["Sam"], gender="male")
    tell(two, ray, "Sam Taurus drives.", facts=[fact("Sam Taurus drives.", ["Sam Taurus"])])
    store = merge_run(conn, same_person(0.9))
    [prompt] = store.graph.active_prompts()
    assert prompt["reason"] == "merge_people"


def test_merge_asks_once_about_a_conflict_the_session_already_found(conn):
    run = accepted_run(conn, "one")
    ray = run.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    jasper = run.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    tell(run, jasper, "Ray broke his arm in 2017.", facts=[fact("Ray Patel broke his arm in 2017.", ["Ray Patel"])])
    first = max(run.graph.facts)
    tell(
        run,
        ray,
        "No, it was 2018.",
        facts=[fact("Ray Patel broke his arm in 2018.", ["speaker"], disputes=f"f{first}")],
    )
    model_conflict = {"facts": ["f2", "f1"], "question": "Did Ray break his arm in 2017 or 2018?"}
    chat = FakeChat({}, merged={"relationships": [], "conflicts": [model_conflict], "same_person": []})

    store = merge_run(conn, chat)

    [prompt] = store.graph.active_prompts()
    assert prompt["reason"] == "merge_conflict" and sorted(prompt["payload"]["fact_ids"]) == sorted(store.graph.facts)


def test_joining_people_closes_only_the_questions_it_answers(conn):
    runs = [accepted_run(conn, name) for name in ("one", "two", "three")]
    ray = runs[0].create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    for run, name in zip(runs, ("Andrew Gao", "Andrew", "Andrew Gall"), strict=True):
        run.show_person(ray)
        run.create_person(name=name, aliases=[], gender="male")
        tell(run, ray, f"{name} climbs.", facts=[fact(f"{name} climbs.", [name])])
    pairs = [{"a": a, "b": b, "confidence": 0.5} for a, b in (("p2", "p3"), ("p3", "p4"))]
    store = merge_run(conn, FakeChat({}, merged={"relationships": [], "conflicts": [], "same_person": pairs}))
    name = {p["id"]: p["name"] for p in store.graph.people.values()}
    [first] = [p for p in store.graph.active_prompts() if name[p["payload"]["keep"]] == "Andrew Gao"]

    resolve_merge(store, first["id"], action="merge_people", fact_id=None)

    [left] = store.graph.active_prompts()
    assert sorted(name[p] for p in left["person_ids"]) == ["Andrew", "Andrew Gall"]


def test_joining_people_closes_a_question_that_now_repeats_another(conn):
    runs = [accepted_run(conn, name) for name in ("one", "two", "three")]
    ray = runs[0].create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    for run, name in zip(runs, ("Sam Torres", "San Torres", "Sam Taurus"), strict=True):
        run.show_person(ray)
        run.create_person(name=name, aliases=[], gender="male")
        tell(run, ray, f"{name} climbs.", facts=[fact(f"{name} climbs.", [name])])
    pairs = [{"a": a, "b": b, "confidence": 0.5} for a, b in (("p2", "p3"), ("p2", "p4"), ("p3", "p4"))]
    store = merge_run(conn, FakeChat({}, merged={"relationships": [], "conflicts": [], "same_person": pairs}))
    name = {p["id"]: p["name"] for p in store.graph.people.values()}
    [first] = [
        p for p in store.graph.active_prompts() if {name[i] for i in p["person_ids"]} == {"Sam Torres", "San Torres"}
    ]

    resolve_merge(store, first["id"], action="merge_people", fact_id=None)

    [left] = store.graph.active_prompts()
    assert "Sam Taurus" in {name[p] for p in left["person_ids"]}
