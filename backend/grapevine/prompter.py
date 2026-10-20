"""The prompt panel: questions the room owes an answer to, and suggested questions to keep people talking."""

from itertools import combinations
from typing import Literal

from grapevine.graph import Record
from grapevine.llm import LLM, Call
from grapevine.store import Store

PROMPT_VERSION = "prompter-v1"
QUESTIONS = ("which_person", "which_speaker", "conflict")
MAX_SHOWN = 2

Trigger = Literal["opener", "pause"]

SYSTEM = """You suggest questions for a group conversation, so the people in the room tell how they know each other and the stories they share. A question is only a suggestion. The room does not need to follow it.

Rules:
- Write at most 2 questions. Each question is one short sentence that names people by first name, for example "Ray and Jasper, how did you two meet?"
- Prefer people and pairs with little talk time or few facts, so a few people do not take over.
- If a fact misses a date, you can ask for it, for example "Sophie, what year was that?"
- Do not repeat a question that is on screen or that the facts already answer.
- reason: opener (a first question between people in the room), pause (the room went quiet), or needs_date (asks for a missing date).
- people: the ids of the people the question is for, for example ["p1", "p2"]."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prompts"],
    "properties": {
        "prompts": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "reason", "people"],
                "properties": {
                    "text": {"type": "string"},
                    "reason": {"type": "string", "enum": ["opener", "pause", "needs_date"]},
                    "people": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def show_prompt(
    store: Store,
    *,
    text: str,
    reason: str,
    person_ids: list[int],
    payload: Record | None = None,
    model_call_id: int | None = None,
) -> Record | None:
    """Adds a prompt. Suggestions make room by replacing older suggestions and never replace a question."""
    active = store.graph.active_prompts()
    if any(p["text"] == text for p in active):
        return None
    if reason not in QUESTIONS and sum(p["reason"] in QUESTIONS for p in active) >= MAX_SHOWN:
        return None
    prompt = store.add_prompt(
        text=text, reason=reason, person_ids=person_ids, payload=payload, model_call_id=model_call_id
    )
    suggestions = sorted((p for p in active if p["reason"] not in QUESTIONS), key=lambda p: p["id"])
    for old in suggestions[: max(0, len(active) + 1 - MAX_SHOWN)]:
        store.dismiss_prompt(old["id"])
    return prompt


def user_message(store: Store, trigger: Trigger, *, until_ms: int) -> str:
    graph = store.graph
    room = sorted({pid for pid in graph.speakers.values() if pid is not None})
    segments = sorted((s for s in graph.segments.values() if s["start_ms"] < until_ms), key=lambda s: s["start_ms"])
    talk_ms = {pid: 0 for pid in room}
    for segment in segments:
        if segment["person_id"] in talk_ms:
            talk_ms[segment["person_id"]] += segment["end_ms"] - segment["start_ms"]
    facts = graph.live_facts()

    def first_name(person_id: int) -> str:
        return graph.people[person_id]["name"].split()[0]

    lines = [f"TRIGGER: {trigger}", "", "PEOPLE IN THE ROOM (talk time, facts about them):"]
    for pid in room:
        count = sum(any(p["person_id"] == pid for p in f["people"]) for f in facts)
        lines.append(f"{graph.ref('p', pid)} {graph.people[pid]['name']}: {talk_ms[pid] // 1000}s, {count} facts")
    lines += ["", "PAIRS IN THE ROOM (shared facts):"]
    for a, b in combinations(room, 2):
        shared = sum({a, b} <= {p["person_id"] for p in f["people"]} for f in facts)
        lines.append(f"{first_name(a)} and {first_name(b)}: {shared}")
    lines += ["", "FACTS WITHOUT A DATE:"]
    lines += [f"- {f['text']}" for f in facts if f["kind"] in ("relationship", "story") and not f["start_date"]][
        -5:
    ] or ["(none)"]
    lines += ["", "RECENT TURNS:"]
    recent = segments[-6:]
    lines += [
        f"{graph.people[s['person_id']]['name'] if s['person_id'] in graph.people else 'unknown'}: {s['text']}"
        for s in recent
    ] or ["(none)"]
    lines += ["", "ON SCREEN:"]
    lines += [f"- {p['text']}" for p in graph.active_prompts()] or ["(none)"]
    return "\n".join(lines)


def prompter_call(store: Store, trigger: Trigger, *, until_ms: int, model: str, seed: int) -> Call | None:
    """Builds the call from the graph as it is now, so the prompter can run in parallel with extraction."""
    if sum(pid is not None for pid in store.graph.speakers.values()) < 2:
        return None
    user = user_message(store, trigger, until_ms=until_ms)
    return Call("prompter", model, PROMPT_VERSION, SYSTEM, user, SCHEMA, seed, max_tokens=300)


async def suggest(store: Store, llm: LLM, call: Call, *, at_ms: int) -> None:
    result = await llm.call(call, run_id=store.run_id, audio_ms=at_ms)
    if result.output is None:
        return
    store.audio_ms = at_ms
    room = [pid for pid in store.graph.speakers.values() if pid is not None]
    for prompt in result.output["prompts"][:MAX_SHOWN]:
        ids = [pid for ref in prompt["people"] if (pid := store.graph.lookup("p", ref)) in room]
        show_prompt(
            store, text=prompt["text"], reason=prompt["reason"], person_ids=ids, model_call_id=result.model_call_id
        )
