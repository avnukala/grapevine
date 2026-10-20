"""The end-of-session pass: a larger model writes stories and relationship summaries from the facts of a run."""

from grapevine.llm import LLM, Call
from grapevine.store import Store

PROMPT_VERSION = "summarizer-v1"

SYSTEM = """You write the record of a group conversation after it ends. You get the people, the events, the relationships, and every fact people told.

Write:
- stories: one story for each event that has facts, and one for each other story that several facts tell together. "text" tells the story in 2 to 4 plain sentences with full names, the place, and the date. "title" is 3 to 6 words. "event" is the event id, or null. "people" lists the ids of everyone in it.
- relationships: for each relationship id, a "summary" of 1 or 2 sentences built from all of its facts, and a "label" of 1 to 3 words.

Use only the facts. Where facts disagree, say so in the text. Skip facts that a later fact revised."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stories", "relationships"],
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["event", "title", "text", "people"],
                "properties": {
                    "event": {"type": ["string", "null"]},
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "people": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "label", "summary"],
                "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "summary": {"type": "string"}},
            },
        },
    },
}


def user_message(store: Store) -> str:
    graph = store.graph
    facts = graph.live_facts()
    lines = ["PEOPLE:"]
    lines += [f"{graph.ref('p', p['id'])} {p['name']}" for p in graph.people.values()]
    lines += ["", "EVENTS:"]
    lines += [
        f"{graph.ref('e', e['id'])} {e['name']} ({e['start_date'] or e['when_text'] or 'no date'})"
        for e in graph.events.values()
    ]
    lines += ["", "RELATIONSHIPS:"]
    lines += [
        f"{graph.ref('r', r['id'])} {graph.ref('p', r['a'])}-{graph.ref('p', r['b'])} {r['category']} "
        f'"{r["label"]}" {r["status"]}: {r["summary"] or ""} '
        f"(facts: {', '.join(graph.ref('f', i) for i in r['evidence']) or 'none'})"
        for r in graph.relationships.values()
    ]
    lines += ["", "FACTS:"]
    lines += [
        f"{graph.ref('f', f['id'])} [{f['kind']}{', event ' + graph.ref('e', f['event_id']) if f['event_id'] else ''}"
        f"{', ' + f['start_date'] if f['start_date'] else ''}{', disputed' if f['disputed'] else ''}] {f['text']}"
        for f in facts
        if f["kind"] != "story" or f["segment_ids"]
    ]
    return "\n".join(lines)


async def summarize(store: Store, llm: LLM, *, model: str, seed: int) -> None:
    if not store.graph.facts:
        return
    call = Call("summarizer", model, PROMPT_VERSION, SYSTEM, user_message(store), SCHEMA, seed, max_tokens=3000)
    result = await llm.call(call, run_id=store.run_id, audio_ms=store.audio_ms)
    if result.output is None:
        return
    graph = store.graph
    with store.batch():
        for story in result.output["stories"]:
            event = graph.events.get(graph.lookup("e", story["event"]))
            people = [pid for ref in story["people"] if (pid := graph.lookup("p", ref)) is not None]
            store.add_fact(
                {
                    "kind": "story",
                    "text": story["title"],
                    "detail": story["text"],
                    "when_text": event["when_text"] if event else None,
                    "start_date": event["start_date"] if event else None,
                    "end_date": event["end_date"] if event else None,
                    "date_precision": event["date_precision"] if event else None,
                    "event_id": event["id"] if event else None,
                },
                people=[(pid, "subject") for pid in people],
                segment_ids=[],
                model_call_id=result.model_call_id,
            )
        for summary in result.output["relationships"]:
            relationship = graph.relationships.get(graph.lookup("r", summary["id"]))
            if relationship is None:
                continue
            store.upsert_relationship(
                a=relationship["a"],
                b=relationship["b"],
                category=relationship["category"],
                fields={"label": summary["label"], "summary": summary["summary"]},
                evidence=[],
                model_call_id=result.model_call_id,
            )
