import pytest

from grapevine.runs import open_store


def test_a_failed_batch_leaves_the_graph_as_the_database_has_it(store):
    store.create_person(name="Marcus Hale", aliases=[], gender="male")
    with pytest.raises(RuntimeError), store.batch():
        store.create_person(name="Tina Alvarez", aliases=[], gender="female")
        store.set_phase("conversation")
        raise RuntimeError

    fresh = open_store(store.conn, store.run_id)
    assert store.graph == fresh.graph
    assert store.seq == fresh.seq == 1


def test_removing_an_event_does_not_bring_back_a_fact_of_a_removed_person(store):
    marcus = store.create_person(name="Marcus Hale", aliases=[], gender="male")["id"]
    tina = store.create_person(name="Tina Alvarez", aliases=[], gender="female")["id"]
    event_id = store.upsert_event(None, {"name": "Lisbon trip"}, people=[marcus], model_call_id=None)["id"]
    kept = store.add_fact(
        {"kind": "story", "text": "Marcus went to Lisbon.", "event_id": event_id},
        people=[(marcus, "subject")],
        segment_ids=[],
        model_call_id=None,
    )["id"]
    store.add_fact(
        {"kind": "story", "text": "Tina went to Lisbon.", "event_id": event_id},
        people=[(tina, "subject")],
        segment_ids=[],
        model_call_id=None,
    )
    store.remove_person(tina)

    store.remove_event(event_id, model_call_id=None)

    assert list(store.graph.facts) == [kept]
    assert store.graph.facts[kept]["event_id"] is None
