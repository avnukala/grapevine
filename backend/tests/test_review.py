from grapevine.review import agreed

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
