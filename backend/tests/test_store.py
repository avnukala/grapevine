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
