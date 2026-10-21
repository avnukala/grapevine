from grapevine.review import agreed, apply_happenings, candidate_groups, group_lines, ties

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
    ]
    assert group_lines(texts, ["Sunday Sends"]) == [0, 1, 2, 4]


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
