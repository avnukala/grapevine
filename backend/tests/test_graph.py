from grapevine.graph import Graph


def test_refs_count_again_after_a_record_is_removed():
    graph = Graph()
    for event_id in (5, 9, 7):
        graph.apply({"type": "event", "event": {"id": event_id, "name": "Party", "people": []}})
    assert graph.ref("e", 7) == "e3"

    graph.apply({"type": "event_removed", "id": 9})

    assert graph.ref("e", 7) == "e2"
