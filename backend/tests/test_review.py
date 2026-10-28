import asyncio

from grapevine.llm import Call, Result
from grapevine.review import (
    agreed,
    apply_groups,
    apply_happenings,
    call_or_retry,
    candidate_groups,
    group_lines,
    hide_small_talk,
    quoted,
    said_in_lines,
    small_talk_message,
    ties,
)

IDS = {"Ray Patel": 1, "Jasper Lin": 2, "Sophie Moreau": 3}


def tie(a, b, category, met_on=None):
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


def test_a_relationship_most_votes_list_is_kept_with_any_date_a_vote_gives():
    votes = [
        [tie("Ray Patel", "Jasper Lin", "friend"), tie("Ray Patel", "Sophie Moreau", "family")],
        [tie("Jasper Lin", "Ray Patel", "friend", "2016")],
        [tie("Sophie Moreau", "Jasper Lin", "romantic")],
    ]
    kept = agreed(votes, lambda name, _: IDS[name], model_call_id=1, needed=2)
    assert [(e["category"], e["met_on"]) for e in kept] == [("friend", "2016")]


def test_group_lines_take_misspelled_names_their_answers_and_the_group_noun():
    texts = [
        "Owen, are you going with the Sunday Sense crew?",
        "Yes. Me, Sam, and Kevin.",
        "Nice.",
        "Who wants tea?",
        "The whole crew is going back.",
        "Me too.",
        "So now it's five of us.",
    ]
    assert group_lines(texts, ["Sunday Sends"]) == [0, 1, 2, 4, 5]
    assert group_lines(texts, ["The Know It Alls"]) == []
    assert group_lines(["I moved to Lisbon.", "Then you joined the Night Hawks?", "Yes."], ["Night Hawks"]) == [0, 1, 2]


def test_a_record_event_takes_the_date_of_the_first_happening_linked_to_it(store):
    megan = store.create_person(name="Megan Walsh", aliases=["Megan"], gender="female")["id"]
    nina = store.create_person(name="Nina Shah", aliases=["Nina"], gender="female")["id"]
    move = store.upsert_event(
        None, {"name": "Move to Denver", "start_date": "2015"}, people=[megan], model_call_id=None
    )
    birthday = store.upsert_event(None, {"name": "Nina's birthday"}, people=[nina], model_call_id=None)

    def happening(name: str, record: str | None, start: str) -> dict:
        return {"name": name, "record": record, "start_date": start, "place": None, "people": []}

    happenings = [
        happening("Megan's move to Denver", "Move to Denver", "2016"),
        happening("Nina's move to Denver", "Move to Denver", "2019"),
        happening("Nina's move to Boston", "Nina's birthday", "2012"),
        happening("...", "Nina's birthday", "2020-03"),
        happening("Megan and Nina's meeting", None, "2020"),
    ]
    record = {"Move to Denver": move["id"], "Nina's birthday": birthday["id"]}
    apply_happenings(store, happenings, record, lambda name, _: None, model_call_id=1)
    assert sorted((e["name"], e["start_date"]) for e in store.graph.events.values()) == [
        ("Move to Denver", "2016"),
        ("Nina's birthday", "2020-03"),
        ("Nina's move to Boston", "2012"),
        ("Nina's move to Denver", "2019"),
    ]


def test_candidate_groups_are_named_groups_that_the_record_does_not_have():
    texts = [
        "The No-It-Alls came in fourth on Tuesday.",
        "Are you going with the Sunday Sense crew?",
        "Matt runs the Dynasty League now. The landlord knows.",
    ]
    assert candidate_groups(texts, ["Sunday Sends"]) == ["The No-It-Alls", "Dynasty League"]


def test_bandmates_are_friends_even_when_the_band_is_paid():
    entry = {"a": "A", "b": "B", "category": "professional", "label": "bandmates", "status": "current"}
    coworker = {**entry, "label": "coworkers"}
    assert [e["category"] for e in ties([{**entry, "in_touch": True}, {**coworker, "in_touch": True}])] == [
        "friend",
        "professional",
    ]


class CutOff:
    """Returns cut-off JSON for the first call and valid JSON after it."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    async def call(self, call: Call, **_) -> Result:
        self.calls.append(call)
        return Result(None if len(self.calls) == 1 else {"members": []}, len(self.calls))


def test_cut_off_output_is_asked_again_with_another_seed_and_temperature(store):
    llm = CutOff()
    result = asyncio.run(call_or_retry(store, llm, Call("reviewer", "m", "v", "s", "u", {}, seed=7)))
    assert result.output == {"members": []}
    assert [(c.seed, c.temperature) for c in llm.calls] == [(7, 0), (10, 0.3)]


def test_a_friend_labelled_coworker_is_professional():
    entry = {"a": "A", "b": "B", "category": "friend", "status": "current", "in_touch": True}
    labels = ["coworker", "Co-worker", "colleague", "manager", "boss", "teammate at work", "teammate"]
    assert [e["category"] for e in ties([{**entry, "label": label} for label in labels])] == [
        *["professional"] * 6,
        "friend",
    ]


def test_a_tie_needs_a_quote_from_the_transcript():
    texts = ["Ray and I were freshman roommates at Cornell, back in 2016.", "Happy birthday!", "We still climb."]
    assert quoted("Ray and I were freshmen roommates at Cornell", texts)
    assert quoted("Happy birthday! We still climb.", texts)
    assert quoted("Happy birthday. Ray and I were freshman roommates at Cornell", texts)
    assert quoted("Jasper Lin: Ray and I were freshman roommates... We still climb.", texts)
    assert not quoted("Ray and I were freshman roommates... They are friends within the group.", texts)
    assert not quoted("They are friends within the group.", texts)
    assert not quoted("...", texts)


def test_a_second_fact_about_a_member_does_not_list_them_twice(store):
    felix = store.create_person(name="Felix Ward", aliases=[], gender="male")["id"]
    group = store.group_for("Night Hawks")
    for text in ["Felix Ward is in the Night Hawks.", "Felix Ward runs the Night Hawks."]:
        fact = store.add_fact({"kind": "membership", "text": text}, people=[], segment_ids=[], model_call_id=None)
        store.add_member(group_id=group["id"], person_id=felix, since=None, until=None, fact_id=fact["id"])
    assert [m["person_id"] for m in store.graph.members.values()] == [felix]


def test_a_removed_member_leaves_the_group_and_the_last_one_takes_the_group(store):
    felix = store.create_person(name="Felix Ward", aliases=[], gender="male")["id"]
    dana = store.create_person(name="Dana Cole", aliases=[], gender="female")["id"]
    group = store.group_for("Night Hawks")
    fact = store.add_fact({"kind": "membership", "text": "x"}, people=[], segment_ids=[], model_call_id=None)
    for pid in (felix, dana):
        store.add_member(group_id=group["id"], person_id=pid, since=None, until=None, fact_id=fact["id"])
    store.remove_member(group_id=group["id"], person_id=felix)
    assert [m["person_id"] for m in store.graph.members.values()] == [dana]
    store.remove_member(group_id=group["id"], person_id=dana)
    assert store.graph.groups == {}
    assert store.conn.execute("SELECT COUNT(*) FROM group_members").fetchone()[0] == 0


def test_the_review_keeps_the_members_lines_name_and_drops_events_kept_as_groups(store):
    ids = {
        n: store.create_person(name=n, aliases=[], gender="female")["id"]
        for n in ["Felix Ward", "Greta", "Mia Rao", "Dana Cole"]
    }
    store.upsert_event(None, {"name": "Harvest Supper 2024"}, people=[], model_call_id=None)
    fact = store.add_fact({"kind": "membership", "text": "x"}, people=[], segment_ids=[], model_call_id=None)
    for group, name in [("Night Hawks", "Felix Ward"), ("Night Hawks", "Greta"), ("Harvest Supper", "Felix Ward")]:
        store.add_member(
            group_id=store.group_for(group)["id"], person_id=ids[name], since=None, until=None, fact_id=fact["id"]
        )
    entry = {
        "name": "Night Hawks",
        "members": ["Felix Ward", "Dana Cole"],
        "named": ["Felix Ward", "Mia Rao", "Dana Cole"],
    }
    apply_groups(store, [entry], lambda name, _: ids[name], model_call_id=1)
    names = {pid: name for name, pid in ids.items()}
    assert [g["name"] for g in store.graph.groups.values()] == ["Night Hawks"]
    assert sorted(names[m["person_id"]] for m in store.graph.members.values()) == ["Dana Cole", "Felix Ward"]


def test_small_talk_is_hidden_and_lasting_facts_stay(store):
    felix = store.create_person(name="Felix Ward", aliases=[], gender="male")["id"]
    segment = store.add_segment(
        start_ms=0,
        end_ms=1,
        speaker_label="A",
        timings={},
        person_id=felix,
        confidence=1.0,
        text="I teach in Lisbon. The soup is cold.",
    )
    texts = [("personal", "Felix Ward teaches in Lisbon."), ("story", "The soup is cold."), ("membership", "x")]
    ids = [
        store.add_fact(
            {"kind": kind, "text": text}, people=[(felix, "subject")], segment_ids=[segment["id"]], model_call_id=None
        )["id"]
        for kind, text in texts
    ]
    user, _, facts = small_talk_message(store)
    assert list(facts.values()) == ids[:2] and "f2 The soup is cold." in user
    answers = {"f1": {"note": "job", "lasting": True}, "f2": {"note": "tonight", "lasting": False}}
    hide_small_talk(store, answers, facts, model_call_id=1)
    assert [f["text"] for f in store.graph.live_facts()] == ["Felix Ward teaches in Lisbon.", "x"]


def test_the_whole_transcript_adds_only_members_the_group_lines_name_or_who_speak_of_themselves():
    texts = ["Felix, you are in the Night Hawks.", "Yes, me, Dana, and Greta.", "Tina and I founded them."]
    speakers = ["Mia Rao", "Felix Ward", "Tina Alvarez"]
    group = {"name": "Night Hawks", "also_called": [], "members": ["Mia Rao", "Dana Cole", "Greta", "Tina Alvarez"]}
    assert said_in_lines(group, texts, speakers) == ["Dana Cole", "Greta", "Tina Alvarez"]
    assert said_in_lines({**group, "name": "Harbor Club"}, texts, speakers) == group["members"]


def test_ties_and_their_evidence_are_not_offered_as_small_talk(store):
    felix = store.create_person(name="Felix Ward", aliases=[], gender="male")["id"]
    dana = store.create_person(name="Dana Cole", aliases=[], gender="female")["id"]
    segment = store.add_segment(
        start_ms=0, end_ms=1, speaker_label="A", timings={}, person_id=felix, confidence=1.0, text="We married."
    )
    texts = [("relationship", "Felix and Dana are a couple."), ("story", "Felix and Dana married."), ("story", "Hi.")]
    ids = [
        store.add_fact({"kind": kind, "text": text}, people=[], segment_ids=[segment["id"]], model_call_id=None)["id"]
        for kind, text in texts
    ]
    store.upsert_relationship(a=felix, b=dana, category="romantic", fields={}, evidence=[ids[1]], model_call_id=None)
    _, _, facts = small_talk_message(store)
    assert list(facts.values()) == [ids[2]]


def test_a_second_person_with_a_members_first_name_does_not_join(store):
    ids = {
        n: store.create_person(name=n, aliases=[], gender="male")["id"] for n in ["Felix Ward", "Felix Hale", "Mia Rao"]
    }
    fact = store.add_fact({"kind": "membership", "text": "x"}, people=[], segment_ids=[], model_call_id=None)
    group = store.group_for("Night Hawks")
    store.add_member(group_id=group["id"], person_id=ids["Felix Ward"], since=None, until=None, fact_id=fact["id"])
    entry = {
        "name": "Night Hawks",
        "members": ["Felix Hale", "Mia Rao"],
        "named": ["Felix Ward", "Felix Hale", "Mia Rao"],
    }
    apply_groups(store, [entry], lambda name, _: ids[name], model_call_id=1)
    names = {pid: name for name, pid in ids.items()}
    assert sorted(names[m["person_id"]] for m in store.graph.members.values()) == ["Felix Ward", "Mia Rao"]
