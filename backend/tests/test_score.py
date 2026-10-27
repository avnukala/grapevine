from grapevine.graph import Graph
from grapevine.score import facts

EXPECTED = {"people": [], "relationships": [], "groups": [], "disputes": 0, "ambiguous_names": []}


def test_an_event_the_room_disputes_counts_with_either_year():
    graph = Graph(events={1: {"name": "Move to Elm Street", "start_date": "2022-08"}})
    settled = {**EXPECTED, "events": [{"name": "Move to Elm Street", "year": 2023}]}
    disputed = {**EXPECTED, "events": [{"name": "Move to Elm Street", "year": 2023, "years": [2022, 2023]}]}

    assert facts(graph, settled)["events"] == "0/1"
    assert facts(graph, disputed)["events"] == "1/1"
