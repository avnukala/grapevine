"""The global graph. A merge run reads the facts of accepted runs, at most one per session, and writes
global relationships. Only conflicts go to manual review: facts that disagree, and people the model cannot
match with high confidence."""

import json
from itertools import combinations
from typing import Literal

from grapevine.actions import ActionError, active_prompt
from grapevine.graph import Graph, Record
from grapevine.llm import LLM, Call
from grapevine.names import misspelled
from grapevine.store import Store

PROMPT_VERSION = "merger-v1"
SAME_PERSON_CONFIDENCE = 0.95

MergeAction = Literal["keep", "keep_both", "remove_edge", "merge_people", "keep_apart"]

SYSTEM = """You merge what several conversations told about a group of friends into one record.

Write:
- relationships: for each relationship ref, a "summary" of 1 or 2 sentences from all its facts, a "label" of 1 to 3 words, and a "status" (current or past).
- conflicts: facts from different sessions that cannot both be true, for example two different years for the same event. Give the fact refs and a short "question" that asks which one is right. Do not list facts that only add detail.
- same_person: one entry for each pair under CANDIDATE PAIRS and no other pair. "a" and "b" are the two ids, and "confidence" from 0 to 1 says how sure you are that they are the same person. Names that differ by a spelling mistake, such as "Sam Taurus" and "Sam Torres", are often the same person. Use the names, the aliases, and what the facts say about them."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relationships", "conflicts", "same_person"],
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "label", "summary", "status"],
                "properties": {
                    "ref": {"type": "string"},
                    "label": {"type": "string"},
                    "summary": {"type": "string"},
                    "status": {"type": "string", "enum": ["current", "past"]},
                },
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["facts", "question"],
                "properties": {"facts": {"type": "array", "items": {"type": "string"}}, "question": {"type": "string"}},
            },
        },
        "same_person": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["a", "b", "confidence"],
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "confidence": {"type": "number"}},
            },
        },
    },
}


def review(store: Store, **prompt) -> None:
    """Adds an item to the merge review. Unlike the live prompt panel, the review shows every item."""
    if not any(p["text"] == prompt["text"] for p in store.graph.active_prompts()):
        store.add_prompt(**prompt)


def canonical(store: Store, person_id: int) -> int:
    while into := store.conn.execute("SELECT merged_into_id FROM people WHERE id = ?", (person_id,)).fetchone()[0]:
        person_id = into
    return person_id


def accepted_graphs(store: Store) -> list[Graph]:
    rows = store.conn.execute("SELECT id FROM runs WHERE accepted = 1 AND kind != 'merge' ORDER BY id").fetchall()
    graphs = []
    for (run_id,) in rows:
        graph = Graph()
        for (op,) in store.conn.execute("SELECT op_json FROM changes WHERE run_id = ? ORDER BY seq", (run_id,)):
            graph.apply(json.loads(op))
        graphs.append(graph)
    return graphs


def copy_records(store: Store, graph: Graph) -> None:
    """Shows the people, events, groups, and facts of an accepted run in the merge run, by canonical person."""

    def people(records: list[Record]) -> list[Record]:
        return [{**p, "person_id": canonical(store, p["person_id"])} for p in records]

    for person_id in graph.people:
        store.show_person(canonical(store, person_id))
    for event in graph.events.values():
        store.emit({"type": "event", "event": {**event, "people": people(event["people"])}})
    for group in graph.groups.values():
        if group["id"] not in store.graph.groups:
            store.emit({"type": "group", "group": group})
    for member in graph.members.values():
        store.emit({"type": "member", "member": {**member, "person_id": canonical(store, member["person_id"])}})
    for fact in graph.facts.values():
        told_by = fact["told_by_person_id"] and canonical(store, fact["told_by_person_id"])
        store.emit({"type": "fact", "fact": {**fact, "people": people(fact["people"]), "told_by_person_id": told_by}})


def combine_relationships(store: Store, graphs: list[Graph]) -> None:
    """Writes one global relationship per pair and category. Later sessions win, and the earliest date they
    met stays. Sessions that disagree on the status or the date they met go to review."""
    combined: dict[tuple[int, int, str], list[Record]] = {}
    for graph in graphs:
        for rel in graph.relationships.values():
            a, b = sorted((canonical(store, rel["a"]), canonical(store, rel["b"])))
            if a != b:
                combined.setdefault((a, b, rel["category"]), []).append(rel)

    for (a, b, category), rels in combined.items():
        latest = rels[-1]
        met = sorted(r["met_on"] for r in rels if r["met_on"])
        merged = store.upsert_relationship(
            a=a,
            b=b,
            category=category,
            fields={
                "label": latest["label"],
                "summary": latest["summary"],
                "status": latest["status"],
                "met_on": met[0] if met else None,
                "met_event_id": latest["met_event_id"],
                "closeness": latest["closeness"],
                "recency": latest["recency"],
                "sentiment": latest["sentiment"],
            },
            evidence=sorted({f for r in rels for f in r["evidence"]}),
            model_call_id=None,
        )
        statuses = {r["status"] for r in rels}
        if len(statuses) > 1 or len({m[:4] for m in met}) > 1:
            names = " and ".join(store.graph.people[p]["name"].split()[0] for p in (a, b))
            review(
                store,
                text=f"Sessions disagree about {names} ({category}): "
                + "; ".join(f"{r['status']}, met {r['met_on'] or '?'}" for r in rels),
                reason="merge_conflict",
                person_ids=[a, b],
                payload={"fact_ids": merged["evidence"], "relationship_ids": [merged["id"]]},
            )


def candidate_pairs(store: Store) -> list[tuple[int, int]]:
    """Pairs of people whose names could mean the same person, such as "Andrew" and "Andrew Gao"."""

    def could_match(x: Record, y: Record) -> bool:
        names_x = {x["name"].lower(), *(a.lower() for a in x["aliases"])}
        names_y = {y["name"].lower(), *(a.lower() for a in y["aliases"])}
        if x["name"].lower() in names_y or y["name"].lower() in names_x:
            return True
        for short, full in ((x, y), (y, x)):
            names = {full["name"].split()[0].lower(), *(a.lower() for a in full["aliases"])}
            if " " not in short["name"] and short["name"].lower() in names:
                return True
        return misspelled(x["name"], y["name"])

    people = list(store.graph.people.values())
    return [(x["id"], y["id"]) for x, y in combinations(people, 2) if could_match(x, y)]


def user_message(store: Store, pairs: list[tuple[int, int]]) -> str:
    graph = store.graph
    lines = ["PEOPLE:"]
    for person in graph.people.values():
        facts = [
            f["text"]
            for f in graph.live_facts()
            if any(p["person_id"] == person["id"] and p["role"] != "teller" for p in f["people"])
        ]
        lines.append(
            f"{graph.ref('p', person['id'])} {person['name']} (aliases: {', '.join(person['aliases'])}): "
            + " ".join(facts[:4])
        )
    lines += ["", "CANDIDATE PAIRS:"]
    lines += [
        f"{graph.ref('p', a)} {graph.people[a]['name']} and {graph.ref('p', b)} {graph.people[b]['name']}"
        for a, b in pairs
    ] or ["(none)"]
    lines += ["", "RELATIONSHIPS:"]
    for rel in graph.relationships.values():
        evidence = "; ".join(graph.facts[f]["text"] for f in rel["evidence"] if f in graph.facts)
        lines.append(
            f"{graph.ref('r', rel['id'])} {graph.people[rel['a']]['name']} - {graph.people[rel['b']]['name']} "
            f'{rel["category"]} "{rel["label"]}" {rel["status"]}: {evidence}'
        )
    lines += ["", "FACTS:"]
    lines += [
        f"{graph.ref('f', f['id'])} {f['text']}" + (f" [{f['start_date']}]" if f["start_date"] else "")
        for f in graph.live_facts()
    ]
    return "\n".join(lines)


def merge_people(store: Store, keep: int, into_other: int) -> None:
    store.conn.execute("UPDATE people SET merged_into_id = ? WHERE id = ?", (keep, into_other))
    store.emit({"type": "person", "person": store.person(into_other)})


async def merge(store: Store, llm: LLM, *, model: str, seed: int) -> None:
    graphs = accepted_graphs(store)
    with store.batch():
        for graph in graphs:
            copy_records(store, graph)
        combine_relationships(store, graphs)
        for graph in graphs:
            for prompt in graph.active_prompts():
                if prompt["reason"] == "conflict":
                    review(
                        store,
                        text=prompt["text"],
                        reason="merge_conflict",
                        person_ids=prompt["person_ids"],
                        payload={"fact_ids": prompt["payload"]["fact_ids"], "relationship_ids": []},
                    )

    pairs = candidate_pairs(store)
    if not store.graph.facts:
        return
    call = Call("merger", model, PROMPT_VERSION, SYSTEM, user_message(store, pairs), SCHEMA, seed, max_tokens=3000)
    result = await llm.call(call, run_id=store.run_id, audio_ms=0)
    if result.output is None:
        return
    graph = store.graph
    with store.batch():
        for item in result.output["relationships"]:
            rel = graph.relationships.get(graph.lookup("r", item["ref"]))
            if rel is not None:
                store.upsert_relationship(
                    a=rel["a"],
                    b=rel["b"],
                    category=rel["category"],
                    fields={"label": item["label"], "summary": item["summary"], "status": item["status"]},
                    evidence=[],
                    model_call_id=result.model_call_id,
                )
        for conflict in result.output["conflicts"]:
            fact_ids = [f for ref in conflict["facts"] if (f := graph.lookup("f", ref)) is not None]
            if len(fact_ids) < 2:
                continue
            relationship_ids = [r["id"] for r in graph.relationships.values() if set(fact_ids) & set(r["evidence"])]
            review(
                store,
                text=conflict["question"],
                reason="merge_conflict",
                person_ids=[],
                payload={"fact_ids": fact_ids, "relationship_ids": relationship_ids},
                model_call_id=result.model_call_id,
            )
        candidates = {frozenset(p) for p in pairs}
        for match in result.output["same_person"]:
            a, b = graph.lookup("p", match["a"]), graph.lookup("p", match["b"])
            if a is None or b is None or frozenset((a, b)) not in candidates:
                continue
            keep, other = sorted((a, b), key=lambda p: -len(graph.people[p]["name"]))
            if match["confidence"] >= SAME_PERSON_CONFIDENCE:
                merge_people(store, keep, other)
                continue
            review(
                store,
                text=f"Is {graph.people[other]['name']} the same person as {graph.people[keep]['name']}?",
                reason="merge_people",
                person_ids=[keep, other],
                payload={"keep": keep, "other": other, "confidence": match["confidence"]},
                model_call_id=result.model_call_id,
            )


def resolve_merge(store: Store, prompt_id: int, *, action: MergeAction, fact_id: int | None) -> None:
    prompt = store.graph.prompts.get(prompt_id)
    if prompt is None or prompt["dismissed_at_ms"] is not None:
        raise ActionError(f"prompt {prompt_id} is not open")
    payload = prompt["payload"]
    with store.batch():
        match action:
            case "keep":
                active_prompt(store, prompt_id, "merge_conflict")
                if fact_id not in payload["fact_ids"]:
                    raise ActionError(f"fact {fact_id} is not part of this conflict")
                for other in payload["fact_ids"]:
                    if other != fact_id:
                        store.update_fact(other, revised_by_fact_id=fact_id)
                store.update_fact(fact_id, disputed=0)
            case "keep_both":
                active_prompt(store, prompt_id, "merge_conflict")
                for fact in payload["fact_ids"]:
                    store.update_fact(fact, disputed=1)
            case "remove_edge":
                active_prompt(store, prompt_id, "merge_conflict")
                if not payload["relationship_ids"]:
                    raise ActionError("this conflict has no edge to remove")
                for relationship_id in payload["relationship_ids"]:
                    store.remove_relationship(relationship_id)
            case "merge_people":
                active_prompt(store, prompt_id, "merge_people")
                merge_people(store, payload["keep"], payload["other"])
            case "keep_apart":
                active_prompt(store, prompt_id, "merge_people")
        store.dismiss_prompt(prompt_id)
