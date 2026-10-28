"""The end-of-session pass: a larger model writes stories and relationship summaries from the facts of a run."""

from grapevine.llm import LLM, Call, watch_on
from grapevine.store import Store

PROMPT_VERSION = "summarizer-v2"

SYSTEM = """You write the record of a group conversation after it ends. You get the people, the events, the relationships, every fact people told, and the transcript.

Write:
- stories: one for each story someone told about the past, where something went wrong, surprised them, or changed things. Skip plans, logistics, and events with no complication. Use the transcript lines of the story, not only its facts.
  "title": 2 to 5 words, the name the group would use, such as "The Seattle Blackout".
  "text": 3 to 6 plain sentences: who was there and their roles, when, where, what went wrong, how it ended. Keep one or two vivid details or exact words, such as Felix Ward waving a wet towel at the smoke.
  "matters": one sentence on why it matters to the group, or "" if nobody said.
  "tellers": ids of the people who told it. If tellers disagree, give each version in the text: "Mia Rao says 2019. Dana Cole says 2018."
  "event": the event id, or null. "people": ids of everyone in it.
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
                "required": ["event", "title", "text", "matters", "tellers", "people"],
                "properties": {
                    "event": {"type": ["string", "null"]},
                    "title": {"type": "string"},
                    "text": {"type": "string"},
                    "matters": {"type": "string"},
                    "tellers": {"type": "array", "items": {"type": "string"}},
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
        f"{graph.ref('e', e['id'])} {e['name']} ({e['start_date'] or e['when_text'] or 'no date'}"
        f"{', ' + e['place'] if e['place'] else ''})"
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
    lines += ["", "TRANSCRIPT:"]
    lines += [
        f"{graph.people[s['person_id']]['name'] if s['person_id'] in graph.people else s['speaker_label']}: {s['text']}"
        for s in graph.segments.values()
    ]
    return "\n".join(lines)


async def summarize(store: Store, llm: LLM, *, model: str, seed: int) -> None:
    if not store.graph.facts:
        return
    call = Call("summarizer", model, PROMPT_VERSION, SYSTEM, user_message(store), SCHEMA, seed, max_tokens=5000)
    watch = watch_on(store, worker="summarizer")
    result = await llm.call(call, run_id=store.run_id, audio_ms=store.audio_ms, watch=watch)
    if result.output is None:
        return
    graph = store.graph
    with store.batch():
        for story in result.output["stories"]:
            event = graph.events.get(graph.lookup("e", story["event"]))
            people = [pid for ref in story["people"] if (pid := graph.lookup("p", ref)) is not None]
            tellers = [pid for ref in story["tellers"] if (pid := graph.lookup("p", ref)) is not None]
            told = f"Told by {' and '.join(graph.people[pid]['name'] for pid in tellers)}." if len(tellers) > 1 else ""
            store.add_fact(
                {
                    "kind": "story",
                    "text": story["title"],
                    "detail": "\n\n".join(part for part in (story["text"], story["matters"], told) if part),
                    "told_by_person_id": tellers[0] if tellers else None,
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
