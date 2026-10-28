import pytest

from grapevine.actions import ActionError, assign_speaker, pick_person, rename_person, settle_conflict
from grapevine.extract import Held, Teller, apply_output, couple_of, respell_people, retell, user_message
from grapevine.runs import open_store
from grapevine.store import name_matches, proper_group_name
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


def test_a_first_name_from_other_sessions_only_is_a_new_person(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    for name in ["Felix Ward", "Felix Shaw"]:
        store.conn.execute("INSERT INTO people (name, aliases, gender) VALUES (?, '[\"Felix\"]', 'male')", (name,))
    teller = turn(store, ray, "Felix came with us.")
    apply(store, teller, facts=[fact("Felix went out with Ray Patel.", ["Felix", "speaker"])])

    assert store.graph.active_prompts() == []
    assert sorted(p["name"] for p in store.graph.people.values()) == ["Felix", "Ray Patel"]


def test_a_near_miss_of_someone_in_this_run_beats_a_person_of_other_sessions(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    for name in ["Gretta", "Tina Alvares"]:
        store.conn.execute("INSERT INTO people (name, gender) VALUES (?, 'female')", (name,))
    apply(store, turn(store, ray, "Greta is here."), facts=[fact("Greta is here.", ["Greta"])])
    apply(store, turn(store, ray, "Tina Alvarez cooks."), facts=[fact("Tina Alvarez cooks.", ["Tina Alvarez"])])
    apply(store, turn(store, ray, "Gretta called."), facts=[fact("Gretta called Ray.", ["Gretta", "speaker"])])
    apply(store, turn(store, ray, "Tina Alvares too."), facts=[fact("Tina Alvares cooks.", ["Tina Alvares"])])

    assert sorted(p["name"].split()[0] for p in store.graph.people.values()) == ["Greta", "Ray", "Tina"]


def test_a_pet_named_in_the_turn_is_not_a_person(store):
    store.conn.execute("INSERT INTO people (name) VALUES ('Pickles')")
    felix = store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")["id"]
    teller = turn(store, felix, "I live in Austin with my dog Pickles.")
    apply(store, teller, facts=[fact("Felix Ward lives with his dog Pickles.", ["speaker", "Pickles"])])

    assert [p["name"] for p in store.graph.people.values()] == ["Felix Ward"]
    [lives] = store.graph.facts.values()
    assert [p["person_id"] for p in lives["people"]] == [felix]


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


def test_a_new_story_marked_as_a_dispute_is_only_a_fact(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    store.create_person(name="Tina Alvarez", aliases=["Tina"], gender="female")
    store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")
    tina_said = "Felix Ward ate Tina Alvarez's lunch every day."
    apply(store, turn(store, marcus, "Felix ate your lunch."), facts=[fact(tina_said, ["Felix Ward", "Tina Alvarez"])])
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, person_named(store, "Tina Alvarez"), "That is true, but Felix drove me to the hospital."),
        facts=[fact("Felix Ward drove Tina Alvarez to the hospital.", ["Felix Ward", "speaker"], disputes=f"f{first}")],
    )

    assert not any(f["disputed"] for f in store.graph.facts.values())
    assert store.graph.active_prompts() == []


def test_retelling_a_turn_closes_its_conflicts_and_drops_relationships_only_it_showed(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    felix = store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")["id"]
    store.create_person(name="Dana Price", aliases=["Dana"], gender="female")
    apply(store, turn(store, marcus, "We met in 2019."), facts=[fact("They met in 2019.", ["speaker", "Felix Ward"])])
    met = max(store.graph.facts)
    teller = turn(store, felix, "My cousin Dana Price met us in 2018.")
    cousin = {
        "a": "speaker",
        "b": "Dana Price",
        "category": "family",
        "label": "cousin",
        "status": "current",
        "met_on": None,
        "met_event": None,
        "closeness": None,
        "recency": None,
        "sentiment": None,
        "summary": "",
        "facts": [0],
    }
    apply(
        store,
        teller,
        facts=[fact("They met in 2018.", ["speaker", "Marcus Hale"], disputes=f"f{met}")],
        relationships=[cousin],
    )
    old = [max(store.graph.facts)]
    assert store.graph.active_prompts() and store.graph.relationships

    retell(store, old, [], model_call_id=None)

    assert store.graph.active_prompts() == []
    assert store.graph.relationships == {}
    assert not store.graph.facts[met]["disputed"]


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
    segment = store.graph.segments[turn(store, ray, "I climb.").segment_id]

    message = user_message(store, segment=segment, speaker_id=ray, today="2025-10-18", context="full")

    assert "- Ray Patel (aliases: Ray) [in the room]" in message
    assert f"{store.graph.ref('f', max(store.graph.facts))} told by Ray Patel: Ray Patel lives in Brooklyn." in message
    assert message.endswith("NEW TURN (speaker: Ray Patel):\nI climb.")


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


def test_a_removed_person_loses_their_ties_and_facts_and_stays_removed(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(
        store,
        turn(store, ray, "Ray and Argit climb on Tuesdays."),
        people=[{"name": "Argit", "aliases": [], "gender": "male"}],
        facts=[fact("Ray and Argit climb on Tuesdays.", ["speaker", "Argit"])],
        relationships=[
            {
                "a": "speaker",
                "b": "Argit",
                "category": "friend",
                "label": "climbing partner",
                "status": "current",
                "facts": [0],
            }
        ],
    )
    argit = person_named(store, "Argit")
    store.remove_person(argit)
    apply(
        store,
        turn(store, ray, "Argit is in Denver."),
        people=[{"name": "Argit", "aliases": [], "gender": "male"}],
        facts=[fact("Argit is in Denver.", ["Argit"])],
    )

    for graph in (store.graph, open_store(store.conn, store.run_id).graph):
        assert list(graph.people) == [ray]
        assert graph.relationships == {}
        assert graph.live_facts() == []


def test_a_removed_person_closes_the_questions_about_them(store):
    gao = store.create_person(name="Andrew Gao", aliases=["Andrew"], gender="male")["id"]
    moy = store.create_person(name="Andrew Moy", aliases=["Andrew"], gender="male")["id"]
    which = store.add_prompt(text="Which Andrew?", reason="which_person", person_ids=[gao, moy])["id"]
    other = store.add_prompt(text="Gao, where do you work?", reason="question", person_ids=[gao])["id"]

    store.remove_person(moy)

    assert [p["id"] for p in store.graph.active_prompts()] == [other]
    assert store.graph.prompts[which]["dismissed_at_ms"] is not None


def test_a_voice_set_by_hand_moves_every_line_of_that_voice(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    jasper = store.create_person(name="Jasper", aliases=[], gender="male")["id"]
    store.set_speaker("spk0", ray, None)
    first, second = turn(store, ray, "I'm Jasper."), turn(store, ray, "I climb.")

    changed = assign_speaker(store, first.segment_id, person_id=jasper, new_name=None, every_line=True, threshold=0.5)

    assert changed == [first.segment_id, second.segment_id]
    assert store.graph.speakers["spk0"] == jasper
    for segment_id in changed:
        assert store.graph.segments[segment_id]["person_id"] == jasper
        assert store.graph.segments[segment_id]["speaker_manual"]


def test_a_line_set_by_hand_keeps_the_voice_of_its_label(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    jasper = store.create_person(name="Jasper", aliases=[], gender="male")["id"]
    store.set_speaker("spk0", ray, None)
    first, second = turn(store, ray, "Hi."), turn(store, ray, "I climb.")

    assert assign_speaker(
        store, second.segment_id, person_id=jasper, new_name=None, every_line=False, threshold=0.5
    ) == [second.segment_id]
    assert store.graph.speakers["spk0"] == ray
    assert store.graph.segments[first.segment_id]["person_id"] == ray


def room_of(store, *names):
    ids = [store.create_person(name=n, aliases=[n.split()[0]], gender="unknown")["id"] for n in names]
    store.emit({"type": "roster", "person_ids": ids})
    return ids


def test_a_lone_name_one_letter_off_is_the_one_room_member_it_fits(store):
    dana, felix = room_of(store, "Dana Cole", "Felix Ward")
    people = [{"name": "Dena", "aliases": [], "gender": "female"}]
    apply(
        store,
        turn(store, felix, "Happy birthday, Dena."),
        people=people,
        facts=[fact("Felix wished Dena well.", ["Dena"])],
    )
    [wish] = store.graph.facts.values()
    assert [p["person_id"] for p in wish["people"] if p["person_id"] != felix] == [dana]
    assert len(store.graph.people) == 2


def test_lone_misspellings_of_room_members_match_them(store):
    priya, hannah, mia, felix = room_of(store, "Priya Nair", "Hannah Kim", "Mia Rao", "Felix Ward")
    for heard, who in [("Pria", priya), ("Anna", hannah), ("Miaely", mia)]:
        apply(store, turn(store, felix, f"{heard} cooks."), facts=[fact(f"{heard} cooks.", [heard])])
        assert who in {p["person_id"] for f in store.graph.facts.values() for p in f["people"]}
    assert len(store.graph.people) == 4


def test_a_lone_name_that_sounds_like_someone_only_mentioned_is_them(store):
    felix = store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")["id"]
    apply(store, turn(store, felix, "Hanna Cole climbs."), facts=[fact("Hanna Cole climbs.", ["Hanna Cole"])])
    apply(store, turn(store, felix, "Me and Ana."), facts=[fact("Ana climbs with Felix.", ["Ana", "speaker"])])
    assert sorted(p["name"] for p in store.graph.people.values()) == ["Felix Ward", "Hanna Cole"]


def test_misspelled_full_names_of_room_members_match_them(store):
    *_, felix = room_of(store, "Sam Torres", "Mia Rao", "Kate Russo", "Felix Ward")
    for heard in ["San Torres", "Sam Taurus", "Miaely Rao", "Kay Russo"]:
        apply(store, turn(store, felix, f"{heard} cooks."), facts=[fact(f"{heard} cooks.", [heard])])
    assert len(store.graph.people) == 4


def test_a_near_miss_of_two_people_asks_the_room(store):
    felix, _, _ = room_of(store, "Felix Ward", "Dana Cole", "Dina Park")
    apply(store, turn(store, felix, "Dena cooks."), facts=[fact("Dena cooks.", ["Dena"])])
    [prompt] = store.graph.active_prompts()
    assert prompt["reason"] == "which_person"


def test_a_real_different_name_stays_its_own_person(store):
    felix, mark = room_of(store, "Felix Ward", "Mark Hale")
    store.update_person(mark, add_aliases=[], gender="male")
    people = [{"name": "Mary", "aliases": [], "gender": "female"}]
    apply(store, turn(store, felix, "Mary cooks."), people=people, facts=[fact("Mary cooks.", ["Mary"])])
    apply(store, turn(store, felix, "Mark sings."), facts=[fact("Mark sings.", ["Mark"])])
    assert sorted(p["name"] for p in store.graph.people.values()) == ["Felix Ward", "Mark Hale", "Mary"]


def test_renaming_a_person_keeps_the_old_name_as_an_alias_and_fixes_their_facts(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    mia = store.create_person(name="Miaely Rao", aliases=["Miaely"], gender="female")["id"]
    told = [fact("Miaely Rao is a nurse.", ["Miaely Rao"]), fact("Miaely and Marcus climb.", ["Miaely Rao", "speaker"])]
    apply(store, turn(store, marcus, "Miaely Rao is a nurse, and we climb."), facts=told)

    tie = {"category": "friend", "label": "climbing partners", "summary": "Miaely Rao and Marcus Hale climb."}
    store.upsert_relationship(a=marcus, b=mia, category="friend", fields=tie, evidence=[], model_call_id=None)

    rename_person(store, mia, "Mia Rao")

    [relationship] = store.graph.relationships.values()
    assert relationship["summary"] == "Mia Rao and Marcus Hale climb."
    person = store.graph.people[mia]
    assert person["name"] == "Mia Rao" and "Miaely Rao" in person["aliases"]
    assert sorted(f["text"] for f in store.graph.facts.values()) == ["Mia Rao is a nurse.", "Mia and Marcus climb."]


def test_a_person_cannot_take_the_name_of_another_person_in_the_run(store):
    store.create_person(name="Mia Rao", aliases=["Mia"], gender="female")
    dana = store.create_person(name="Dana Cole", aliases=["Dana"], gender="female")["id"]

    with pytest.raises(ActionError):
        rename_person(store, dana, "mia rao")


def test_news_from_the_same_teller_does_not_hide_an_unrelated_fact(store):
    dana = store.create_person(name="Dana Cole", aliases=["Dana"], gender="female")["id"]
    apply(
        store,
        turn(store, dana, "Tina lives in Austin."),
        facts=[fact("Tina Alvarez lives in Austin.", ["Tina Alvarez"])],
    )
    home = max(store.graph.facts)
    apply(store, turn(store, dana, "Tina is a teacher."), facts=[fact("Tina Alvarez is a teacher.", ["Tina Alvarez"])])
    job = max(store.graph.facts)

    news = fact("Tina Alvarez is moving to Denver next spring.", ["Tina Alvarez"], kind="story", revises=f"f{home}")
    apply(store, turn(store, dana, "She is moving to Denver next spring."), facts=[news])
    fix = fact("Tina Alvarez is a professor.", ["Tina Alvarez"], kind="correction", revises=f"f{job}")
    apply(store, turn(store, dana, "Sorry, a professor."), facts=[fix])

    assert store.graph.facts[home]["revised_by_fact_id"] is None
    assert store.graph.facts[job]["revised_by_fact_id"] is not None


def test_a_turn_read_again_does_not_see_its_own_facts(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    first = turn(store, marcus, "I'm Marcus, a nurse in Denver.")
    apply(store, first, facts=[fact("Marcus Hale is a nurse in Denver.", ["speaker"])])
    later = turn(store, marcus, "I climb on Sundays.")

    again = user_message(
        store, segment=store.graph.segments[first.segment_id], speaker_id=marcus, today="2025-10-18", context="full"
    )
    after = user_message(
        store, segment=store.graph.segments[later.segment_id], speaker_id=marcus, today="2025-10-18", context="full"
    )

    assert "nurse in Denver." not in again.split("NEW TURN")[0]
    assert "nurse in Denver." in after


def test_two_people_with_one_full_name_match_as_the_first(store):
    first = store.create_person(name="Mia Rao", aliases=["Mia"], gender="female")["id"]
    store.conn.execute("INSERT INTO people (name, aliases, gender) VALUES ('Mia Rao', '[\"Mia\"]', 'female')")

    assert [p["id"] for p in name_matches(store.conn, "Mia Rao")] == [first]
    assert [p["id"] for p in name_matches(store.conn, "Mia")] == [first]


def test_a_respelling_never_takes_the_name_of_another_person(store):
    store.create_person(name="Mia Rao", aliases=["Mia"], gender="female")
    heard = store.create_person(name="Mira Rao", aliases=["Mira"], gender="female")["id"]

    store.update_person(heard, add_aliases=["Mira Rao"], gender=None, name="Mia Rao")

    person = store.person(heard)
    assert person["name"] == "Mira Rao" and "Mia Rao" in person["aliases"]
    with pytest.raises(ActionError):
        rename_person(store, heard, "Mia Rao")


def test_a_retelling_that_shares_one_word_is_not_a_dispute(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    tina = store.create_person(name="Tina Alvarez", aliases=["Tina"], gender="female")["id"]
    apply(
        store,
        turn(store, marcus, "Tina's kitchen caught fire."),
        facts=[fact("Tina Alvarez's kitchen caught fire in 2021.", ["Tina Alvarez"])],
    )
    first = max(store.graph.facts)
    said = "Tina Alvarez put a burning towel in the sink during the fire."
    apply(
        store, turn(store, tina, "I put the towel in the sink."), facts=[fact(said, ["speaker"], disputes=f"f{first}")]
    )
    apply(
        store,
        turn(store, tina, "It caught fire."),
        facts=[fact("Tina Alvarez's kitchen caught fire in 2021.", ["speaker"], disputes=f"f{first}")],
    )

    assert not any(f["disputed"] for f in store.graph.facts.values())
    assert store.graph.active_prompts() == []


def test_a_dispute_of_a_fact_with_no_teller_stays_a_dispute(store):
    ray = store.create_person(name="Ray Patel", aliases=["Ray"], gender="male")["id"]
    apply(
        store,
        turn(store, None, "Ray moved to Lisbon in 2017."),
        facts=[fact("Ray Patel moved to Lisbon in 2017.", ["Ray Patel"])],
    )
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, ray, "No, 2018."),
        facts=[fact("Ray Patel moved to Lisbon in 2018.", ["speaker"], start_date="2018", disputes=f"f{first}")],
    )

    assert store.graph.facts[first]["revised_by_fact_id"] is None
    assert store.graph.facts[first]["disputed"] and store.graph.facts[max(store.graph.facts)]["disputed"]


def test_two_stories_at_one_place_are_not_a_dispute(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    tina = store.create_person(name="Tina Alvarez", aliases=["Tina"], gender="female")["id"]
    dorm = "Marcus Hale and Tina Alvarez were in the same dorm at the University of Lisbon."
    apply(store, turn(store, marcus, "Same dorm."), facts=[fact(dorm, ["speaker", "Tina Alvarez"], start_date="2018")])
    first = max(store.graph.facts)
    spot = "Tina Alvarez stole Marcus Hale's parking spot during their first week at the University of Lisbon."
    apply(store, turn(store, tina, "I stole his spot."), facts=[fact(spot, ["speaker"], disputes=f"f{first}")])

    assert store.graph.active_prompts() == []


def test_retelling_a_turn_keeps_a_dispute_that_its_new_facts_still_make(store):
    marcus = store.create_person(name="Marcus Hale", aliases=["Marcus"], gender="male")["id"]
    felix = store.create_person(name="Felix Ward", aliases=["Felix"], gender="male")["id"]
    said = "We met at band camp in 2014."
    apply(
        store,
        turn(store, marcus, said),
        facts=[fact("They met at band camp in 2014.", ["speaker", "Felix Ward"], start_date="2014")],
    )
    first = max(store.graph.facts)
    apply(
        store,
        turn(store, None, "No, summer 2015."),
        facts=[
            fact(
                "The speaker met Marcus Hale in 2015.",
                ["speaker", "Marcus Hale"],
                start_date="2015",
                disputes=f"f{first}",
            )
        ],
    )
    old = max(store.graph.facts)
    assert store.graph.active_prompts()
    apply(
        store,
        turn(store, felix, "No, summer 2015."),
        facts=[fact("Felix Ward met Marcus Hale in summer 2015.", ["speaker", "Marcus Hale"], start_date="2015")],
    )
    new = max(store.graph.facts)

    retell(store, [old], [new], model_call_id=None)

    [conflict] = store.graph.active_prompts()
    assert conflict["payload"]["fact_ids"] == [first, new]
    assert store.graph.facts[first]["disputed"] and store.graph.facts[new]["disputed"]
