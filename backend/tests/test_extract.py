from grapevine.actions import pick_person, settle_conflict
from grapevine.extract import Held, Teller, apply_output, couple_of, respell_people, user_message
from grapevine.store import proper_group_name
from tests.fakes import empty_output, fact


def turn(store, speaker_id, text):
    segment = store.add_segment(
        start_ms=len(store.graph.segments) * 1000,
        end_ms=len(store.graph.segments) * 1000 + 900,
        speaker_label="spk0",
        person_id=speaker_id,
        confidence=None,
        text=text,
        timings={},
    )
    return Teller(segment["id"], speaker_id, 1.0, model_call_id=None)


def apply(store, teller, **output):
    apply_output(store, Held({**empty_output(), **output}, teller))


def person_named(store, name):
    return next(p["id"] for p in store.graph.people.values() if p["name"] == name)


def test_new_people_and_the_speaker_get_linked_to_the_fact(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    teller = turn(store, ray, "My girlfriend Priya lives in Lisbon.")
    apply(
        store,
        teller,
        people=[{"name": "Priya", "aliases": [], "gender": "female"}],
        facts=[fact("Priya lives in Lisbon.", ["Priya"])],
        relationships=[
            {
                "a": "speaker",
                "b": "Priya",
                "category": "romantic",
                "label": "girlfriend",
                "status": "current",
                "met_on": None,
                "met_event": None,
                "closeness": 5,
                "recency": 5,
                "sentiment": 2,
                "summary": "Ray Patel dates Priya.",
                "facts": [0],
            }
        ],
    )

    priya = person_named(store, "Priya")
    [stored_fact] = store.graph.facts.values()
    assert {p["person_id"]: p["role"] for p in stored_fact["people"]} == {priya: "subject", ray: "teller"}
    assert stored_fact["told_by_person_id"] == ray
    [relationship] = store.graph.relationships.values()
    assert (relationship["category"], relationship["evidence"]) == ("romantic", [stored_fact["id"]])
    assert store.graph.people[priya]["gender"] == "female"


def test_a_pair_keeps_one_relationship_per_category(store):
    jasper = store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    store.create_person(name="Sophie Moreau", aliases=["Sophie"], gender="female")
    teller = turn(store, jasper, "Sophie and I dated, now we are friends.")
    base = {
        "met_on": None,
        "met_event": None,
        "closeness": None,
        "recency": None,
        "sentiment": None,
        "summary": "",
        "facts": [],
    }
    apply(
        store,
        teller,
        relationships=[
            {**base, "a": "speaker", "b": "Sophie Moreau", "category": "romantic", "label": "ex", "status": "past"},
            {
                **base,
                "a": "Sophie Moreau",
                "b": "speaker",
                "category": "friend",
                "label": "friend",
                "status": "current",
            },
        ],
    )
    apply(
        store,
        teller,
        relationships=[
            {
                **base,
                "a": "speaker",
                "b": "Sophie",
                "category": "friend",
                "label": "good friend",
                "status": "current",
                "closeness": 4,
            },
        ],
    )

    relationships = sorted(store.graph.relationships.values(), key=lambda r: r["category"])
    assert [(r["category"], r["status"], r["label"]) for r in relationships] == [
        ("friend", "current", "good friend"),
        ("romantic", "past", "ex"),
    ]
    assert relationships[0]["closeness"] == 4


def test_ambiguous_first_name_waits_for_the_room(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    gao = store.create_person(name="Andrew Gao", aliases=["Andrew", "Gao"], gender="male")["id"]
    store.create_person(name="Andrew Moy", aliases=["Andrew", "Moy"], gender="male")
    teller = turn(store, ray, "Andrew came with us to Electric Zoo.")
    apply(
        store,
        teller,
        facts=[
            fact("Ray Patel went to Electric Zoo.", ["speaker"]),
            fact("Andrew Gao went to Electric Zoo with Ray Patel.", ["Andrew Gao", "speaker"]),
        ],
    )

    assert len(store.graph.facts) == 1
    [prompt] = store.graph.active_prompts()
    assert prompt["text"] == "Which Andrew: Andrew Gao or Andrew Moy?"

    pick_person(store, prompt["id"], person_id=gao, new_name=None)

    assert store.graph.active_prompts() == []
    assert len(store.graph.facts) == 2
    waited = max(store.graph.facts.values(), key=lambda f: f["id"])
    assert {p["person_id"] for p in waited["people"]} == {gao, ray}


def test_full_name_is_not_ambiguous(store):
    jasper = store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    store.create_person(name="Andrew Gao", aliases=["Andrew", "Gao"], gender="male")
    store.create_person(name="Andrew Moy", aliases=["Andrew", "Moy"], gender="male")
    teller = turn(store, jasper, "Andrew Gao is my coworker.")
    apply(store, teller, facts=[fact("Andrew Gao is Jasper Lin's coworker.", ["Andrew Gao", "speaker"])])
    assert store.graph.active_prompts() == []
    assert len(store.graph.facts) == 1


def test_contradicting_another_teller_is_a_dispute_and_own_correction_settles_it(store):
    sophie = store.create_person(name="Sophie Moreau", aliases=["Sophie"], gender="female")["id"]
    jasper = store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    apply(store, turn(store, sophie, "We met in 2019."), facts=[fact("They met in 2019.", ["speaker", "Jasper Lin"])])
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, jasper, "No, it was 2018."),
        facts=[fact("They met in 2018.", ["speaker", "Sophie Moreau"], kind="correction", revises=f"f{first}")],
    )
    second = max(store.graph.facts)

    assert store.graph.facts[first]["disputed"] and store.graph.facts[second]["disputed"]
    assert store.graph.facts[first]["revised_by_fact_id"] is None
    [conflict] = store.graph.active_prompts()
    assert conflict["reason"] == "conflict"

    apply(
        store,
        turn(store, sophie, "Okay, fine, it was 2018."),
        facts=[fact("They met in 2018.", ["speaker", "Jasper Lin"], kind="correction", revises=f"f{first}")],
    )

    assert store.graph.facts[first]["revised_by_fact_id"] == max(store.graph.facts)
    assert not store.graph.facts[second]["disputed"]
    assert store.graph.active_prompts() == []


def test_settling_a_conflict_keeps_one_fact(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    jasper = store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    apply(
        store,
        turn(store, jasper, "Ray broke his arm in 2017."),
        facts=[fact("Ray broke his arm in 2017.", ["Ray Patel"])],
    )
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, ray, "It was 2018."),
        facts=[fact("Ray broke his arm in 2018.", ["speaker"], disputes=f"f{first}")],
    )
    second = max(store.graph.facts)
    [conflict] = store.graph.active_prompts()

    settle_conflict(store, conflict["id"], keep_fact_id=second)

    assert store.graph.facts[first]["revised_by_fact_id"] == second
    assert not store.graph.facts[second]["disputed"]
    assert [f["id"] for f in store.graph.live_facts()] == [second]


def test_membership_adds_group_members_and_events_take_the_year_from_their_name(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    teller = turn(store, ray, "Maya and I are the Tuesday climbing crew. We went to Electric Zoo 2018.")
    apply(
        store,
        teller,
        events=[
            {
                "ref": "Electric Zoo 2018",
                "name": "Electric Zoo 2018",
                "when_text": None,
                "start_date": None,
                "end_date": None,
                "place": None,
                "people": ["speaker", "Maya"],
            }
        ],
        facts=[
            fact(
                "Ray Patel and Maya are the Tuesday climbing crew.",
                ["speaker", "Maya"],
                kind="membership",
                group="Tuesday climbing crew",
            ),
            fact(
                "Ray Patel and Maya went to Electric Zoo.", ["speaker", "Maya"], kind="story", event="Electric Zoo 2018"
            ),
        ],
    )

    [group] = store.graph.groups.values()
    assert group["name"] == "Tuesday climbing crew"
    assert {m["person_id"] for m in store.graph.members.values()} == {ray, person_named(store, "Maya")}
    [event] = store.graph.events.values()
    assert (event["start_date"], event["date_precision"]) == ("2018", "year")
    assert store.graph.facts[max(store.graph.facts)]["event_id"] == event["id"]


def test_prompt_lists_people_by_name_and_facts_by_id(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    store.set_speaker("spk0", ray, None)
    teller = turn(store, ray, "I live in Brooklyn.")
    apply(store, teller, facts=[fact("Ray Patel lives in Brooklyn.", ["speaker"])])
    segment = store.graph.segments[teller.segment_id]

    message = user_message(store, segment=segment, speaker_id=ray, today="2025-10-18", context="full")

    assert "- Ray Patel (aliases: Ray) [in the room]" in message
    assert f"{store.graph.ref('f', max(store.graph.facts))} told by Ray Patel: Ray Patel lives in Brooklyn." in message
    assert message.endswith("NEW TURN (speaker: Ray Patel):\nI live in Brooklyn.")


def test_an_unknown_speaker_can_dispute_a_fact(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(store, turn(store, ray, "I broke my arm in 2017."), facts=[fact("Ray broke his arm in 2017.", ["speaker"])])
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, None, "No, 2018."),
        facts=[fact("Ray broke his arm in 2018.", ["Ray Patel"], disputes=f"f{first}")],
    )
    [conflict] = store.graph.active_prompts()
    assert conflict["text"].startswith("People disagree.")


def test_a_full_name_completes_a_person_known_by_first_name(store):
    hannah = store.create_person(name="Hannah Kim", aliases=["Hannah"], gender="female")["id"]
    apply(
        store,
        turn(store, hannah, "Tyler proposed."),
        facts=[fact("Tyler proposed to Hannah Kim.", ["Tyler", "speaker"])],
    )
    tyler = person_named(store, "Tyler")
    apply(
        store,
        turn(store, hannah, "Tyler Grant does not climb."),
        facts=[fact("Tyler Grant does not climb.", ["Tyler Grant"])],
    )

    assert store.graph.people[tyler]["name"] == "Tyler Grant"
    assert store.graph.people[tyler]["aliases"] == ["Tyler"]
    assert store.graph.active_prompts() == []


def test_names_that_are_not_names_add_no_person(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(
        store,
        turn(store, ray, "The girl next door helped."),
        facts=[fact("The girl next door helped Ray.", ["the girl next door", "Megan_in_the_room", "speaker"])],
    )
    assert [p["name"] for p in store.graph.people.values()] == ["Ray Patel"]


def test_an_event_named_after_a_group_is_skipped(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(
        store,
        turn(store, ray, "We are the Tuesday climbing crew."),
        events=[
            {
                "ref": "Tuesday Climbing Crew",
                "name": "Tuesday Climbing Crew",
                "when_text": None,
                "start_date": None,
                "end_date": None,
                "place": None,
                "people": ["speaker"],
            }
        ],
        facts=[
            fact("Ray is in the Tuesday climbing crew.", ["speaker"], kind="membership", group="Tuesday climbing crew")
        ],
    )
    assert store.graph.events == {}
    assert len(store.graph.groups) == 1


def test_a_fact_makes_the_event_it_names_when_the_output_does_not_list_it(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(
        store,
        turn(store, ray, "We saw the Lumineers in 2019."),
        facts=[
            fact("Ray Patel saw the Lumineers.", ["speaker"], kind="story", event="Lumineers show", start_date="2019"),
            fact("Ray Patel danced.", ["speaker"], kind="story", event="lumineers show"),
        ],
    )
    [event] = store.graph.events.values()
    assert (event["name"], event["start_date"]) == ("Lumineers show", "2019")
    assert {f["event_id"] for f in store.graph.facts.values()} == {event["id"]}


def test_a_first_name_means_the_person_this_run_talked_about(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    store.conn.execute("INSERT INTO people (name, aliases) VALUES ('Andrew Walsh', '[\"Andrew\"]')")
    apply(store, turn(store, ray, "Andrew Gao is a doctor."), facts=[fact("Andrew Gao is a doctor.", ["Andrew Gao"])])
    apply(store, turn(store, ray, "Andrew came with us."), facts=[fact("Andrew came with Ray.", ["Andrew", "speaker"])])

    assert store.graph.active_prompts() == []
    assert {p["person_id"] for p in store.graph.facts[max(store.graph.facts)]["people"]} == {
        person_named(store, "Andrew Gao"),
        ray,
    }


def test_a_relationship_takes_the_date_of_a_fact_about_how_the_pair_met(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")
    apply(
        store,
        turn(store, ray, "Jasper and I met in 2016."),
        facts=[fact("Ray Patel and Jasper Lin met at Cornell.", ["speaker", "Jasper Lin"], start_date="2016")],
    )
    apply(
        store,
        turn(store, ray, "He is my best friend."),
        relationships=[
            {
                "a": "speaker",
                "b": "Jasper Lin",
                "category": "friend",
                "label": "best friend",
                "status": "current",
                "met_on": None,
                "met_event": None,
                "closeness": 5,
                "recency": 5,
                "sentiment": 2,
                "summary": "",
                "facts": [],
            }
        ],
    )
    [rel] = store.graph.relationships.values()
    assert rel["met_on"] == "2016"


def test_a_bare_first_name_that_fits_two_people_asks_the_room(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    store.create_person(name="Kate Russo", aliases=["Kate"], gender="female")
    store.create_person(name="Kate Miller", aliases=["Kate"], gender="female")
    apply(store, turn(store, ray, "Does Megan know Kate is coming?"))
    [prompt] = store.graph.active_prompts()
    assert prompt["reason"] == "which_person"
    assert prompt["payload"]["mention"] == "Kate"


def test_a_misspelled_full_name_is_the_person_already_known(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    ravi = store.create_person(name="Ravi Kapoor", aliases=["Ravi"], gender="male")["id"]
    apply(store, turn(store, ray, "Rav Kapoor cooks."), facts=[fact("Rav Kapoor cooks.", ["Rav Kapoor"])])
    [cooks] = store.graph.facts.values()
    assert ravi in {p["person_id"] for p in cooks["people"]}


def test_only_a_name_of_one_group_makes_a_group():
    assert proper_group_name("second chapter")
    assert proper_group_name("The Dynasty League")
    assert not proper_group_name("the whole climbing crew")
    assert not proper_group_name("house on Pearl Street")


def test_a_pair_keeps_the_earliest_date_they_met(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    jasper = store.create_person(name="Jasper Lin", aliases=["Jasper"], gender="male")["id"]
    for met_on in ["2016", "2020", "2015"]:
        store.upsert_relationship(
            a=ray, b=jasper, category="friend", fields={"met_on": met_on}, evidence=[], model_call_id=None
        )
    assert [r["met_on"] for r in store.graph.relationships.values()] == ["2015"]


def test_a_turn_respells_a_known_person_whom_whisper_misheard(store):
    kay = store.create_person(name="Kay Russo", aliases=["Kay"], gender="female")["id"]
    store.create_person(name="Kate Miller", aliases=["Kate"], gender="female")
    for text in ["My coworker Kay Russo is coming.", "Kate Russo, not Kate Miller.", "Does Megan know Kate is coming?"]:
        store.add_segment(
            start_ms=0, end_ms=1000, speaker_label="spk0", person_id=None, confidence=0.9, text=text, timings={}
        )
        respell_people(store, text)
    assert store.person(kay)["name"] == "Kate Russo"
    assert "Kay Russo" in store.person(kay)["aliases"]


def test_a_wedding_named_after_two_people_makes_them_a_couple(store):
    arjun = store.create_person(name="Arjun Mehta", aliases=["Arjun"], gender="male")["id"]
    claire = store.create_person(name="Claire Dunn", aliases=["Claire"], gender="female")["id"]
    store.create_person(name="Kevin Doyle", aliases=["Kevin"], gender="male")
    for name in ["Kevin's cousin's wedding", "Arjun Mehta and Claire Dunn's wedding"]:
        couple_of(store, store.upsert_event(None, {"name": name}, people=[], model_call_id=None), model_call_id=None)
    ties = [(r["a"], r["b"], r["category"], r["label"]) for r in store.graph.relationships.values()]
    assert ties == [(arjun, claire, "romantic", "married")]


def test_a_name_that_the_model_writes_with_an_added_ly_loses_it(store):
    ben = store.create_person(name="Ben Carter", aliases=["Ben"], gender="male")["id"]
    apply(
        store,
        turn(store, ben, "Danny Cruz plays drums."),
        facts=[fact("Dannyly Cruz plays drums.", ["Dannyly Cruz"])],
    )
    assert sorted(p["name"] for p in store.graph.people.values()) == ["Ben Carter", "Danny Cruz"]
