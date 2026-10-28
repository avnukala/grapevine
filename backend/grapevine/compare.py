"""Compares two runs on the same session: what each found that the other did not. `judge.py` asks a model
which run is better."""

from grapevine.graph import Graph, Record


def summary(graph: Graph) -> dict[str, set[str]]:
    """The findings of a run, keyed by names so that two runs with different ids compare."""

    def name(person_id: int | None) -> str:
        return graph.people[person_id]["name"] if person_id in graph.people else "unknown"

    return {
        "people": {p["name"] for p in graph.people.values() if p["merged_into_id"] is None},
        "relationships": {
            " - ".join(sorted((name(r["a"]), name(r["b"]))))
            + f": {r['category']} {r['status']}"
            + (f", met {r['met_on']}" if r["met_on"] else "")
            for r in graph.relationships.values()
        },
        "events": {f"{e['name']} ({e['start_date'] or 'no date'})" for e in graph.events.values()},
        "groups": {
            f"{g['name']}: "
            + ", ".join(sorted(name(m["person_id"]) for m in graph.members.values() if m["group_id"] == g["id"]))
            for g in graph.groups.values()
        },
        "facts": {f["text"] for f in graph.live_facts() if f["segment_ids"]},
    }


def differences(a: Graph, b: Graph) -> Record:
    first, second = summary(a), summary(b)
    return {
        key: {"only_a": sorted(first[key] - second[key]), "only_b": sorted(second[key] - first[key])} for key in first
    }
