from grapevine.compare import differences
from grapevine.graph import Graph


def graph(*relationships):
    result = Graph()
    for pid, name in [(1, "Ray Patel"), (2, "Jasper Lin")]:
        result.apply(
            {
                "type": "person",
                "person": {"id": pid, "name": name, "aliases": [], "gender": "male", "merged_into_id": None},
            }
        )
    for rid, (category, met_on) in enumerate(relationships):
        result.apply(
            {
                "type": "relationship",
                "relationship": {
                    "id": rid,
                    "a": 1,
                    "b": 2,
                    "category": category,
                    "label": "",
                    "summary": None,
                    "status": "current",
                    "met_on": met_on,
                    "met_event_id": None,
                    "closeness": None,
                    "recency": None,
                    "sentiment": None,
                    "evidence": [],
                },
            }
        )
    return result


def test_differences_compare_runs_by_name():
    diff = differences(graph(("friend", "2016")), graph(("friend", "2015"), ("professional", None)))
    assert diff["people"] == {"only_a": [], "only_b": []}
    assert diff["relationships"]["only_a"] == ["Jasper Lin - Ray Patel: friend current, met 2016"]
    assert diff["relationships"]["only_b"] == [
        "Jasper Lin - Ray Patel: friend current, met 2015",
        "Jasper Lin - Ray Patel: professional current",
    ]
