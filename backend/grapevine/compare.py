"""Compares two runs on the same session: what each found that the other did not, and optionally a local
model's judgment of which run captured the conversation better."""

from grapevine.graph import Graph, Record
from grapevine.llm import LLM, Call

PROMPT_VERSION = "judge-v1"

SYSTEM = """You compare two knowledge graphs that two runs extracted from the same conversation among friends. Judge which run captured the people, relationships, dates, events, and groups more correctly and completely. A wrong fact counts against a run more than a missing one. Answer with the better run ("A", "B", or "tie") and up to 5 short reasons."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["better", "reasons"],
    "properties": {
        "better": {"type": "string", "enum": ["A", "B", "tie"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
}


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


def judge_call(a: Graph, b: Graph, *, model: str, seed: int) -> Call:
    def block(label: str, graph: Graph) -> str:
        found = summary(graph)
        return f"RUN {label}\n" + "\n".join(f"{key.upper()}:\n- " + "\n- ".join(sorted(v)) for key, v in found.items())

    transcript = "\n".join(
        f"{a.people[s['person_id']]['name'] if s['person_id'] in a.people else 'unknown'}: {s['text']}"
        for s in sorted(a.segments.values(), key=lambda s: s["start_ms"])
    )
    user = f"TRANSCRIPT (from run A):\n{transcript}\n\n{block('A', a)}\n\n{block('B', b)}"
    return Call("judge", model, PROMPT_VERSION, SYSTEM, user, SCHEMA, seed, max_tokens=600)


async def judge(llm: LLM, a: Graph, b: Graph, *, run_id: int, model: str, seed: int) -> Record | None:
    result = await llm.call(judge_call(a, b, model=model, seed=seed), run_id=run_id, audio_ms=0)
    return result.output
