import asyncio

from grapevine.llm import Call, Result
from grapevine.review import agreed, apply_happenings, call_or_retry, candidate_groups, group_lines, quoted, ties

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
