"""Scores a run against a gold checklist with a local model: recall of each gold item, precision of each recorded
fact and relationship, and a 1 to 5 rubric. Two runs on the same sample also compare dimension by dimension, asked
in both orders, so that the position of a run in the prompt does not decide the verdict."""

import asyncio
import json
from dataclasses import dataclass

from grapevine.graph import Graph, Record
from grapevine.llm import LLM, Call

PROMPT_VERSION = "gold-judge-v3"
# The rubric has its own version, so a change to it keeps the cached recall and precision calls.
RUBRIC_VERSION = "gold-rubric-v4"
PRECISION_BATCH = 12
PARALLEL = 2

DIMENSIONS = ("people", "relationships", "timeline", "stories", "groups", "faithfulness", "usefulness")
RUBRIC = """1. people: everyone who matters, with the right name, no duplicates, no made-up people, and in-room people told apart from mentioned ones.
2. relationships: right pairs, type, and status (current or past), with how and when they met, and a specific label ("college roommates", not "friend").
3. timeline: right dates, relative dates resolved from the session date, right order, disputed dates kept as disputes, and no date more precise than the conversation gives.
4. stories: each story has who, what, when, where, what went wrong or was surprising, the outcome, and why it matters, and its facts are tied to one event.
5. groups: named groups with the right members.
6. faithfulness: no made-up, wrong, or misattributed facts. The right teller. Disagreements kept as disputes, not merged.
7. usefulness: would a friend reading it feel it truly captures the group's shared history? Signal over noise, no trivia, no repeats. A 5 means a friend could retell how the group formed, its top stories, and where each person is now. Missing how two people met costs more than missing a pet.
The transcript comes from speech recognition. A name spelled as the transcript spells it is not an error.
Anchors: 5 = a careful human note-taker would agree with all of it. 3 = the main points are right, with clear gaps or a few errors. 1 = mostly missing or wrong."""

RECALL_SYSTEM = """You check whether a record that software wrote about a conversation among friends captures each item of a gold checklist. The gold items are true. For each gold item, find the record items that state the same thing, in any section and in any words.

- "yes": the record states every main point of the gold item, with the right people and, where the gold gives one, the right date or value.
- "partly": the record states some of it, or states it with a wrong or missing date, person, or detail.
- "no": the record does not state it.
Judge the substance only. A misspelled name ("Jasper Lind" for "Jasper Lin"), other words for the same label, or a missing group type is still "yes". A gold item with "disputed": true or a dispute with several claims needs every claim kept apart. One merged claim is "partly". A date more precise than the gold "precision" or "raw" phrase is "partly". For "not_facts", answer "yes" when the record states that small talk or joke as a fact, and "no" when it does not.
List the ids of the record items that support your verdict as evidence. Write the note first, in at most 15 words."""

PRECISION_SYSTEM = """You check the record that software wrote about a conversation among friends against the transcript. For each record item, decide whether the transcript supports it.

- "supported": the transcript says it, with the right people and values. Plain paraphrase and resolving a relative date from the session date are fine.
- "partly": the main point is right, but a detail, date, label, or person is wrong or missing.
- "unsupported": the transcript does not say it, or the item is far more specific than the transcript. A date more precise than the conversation gives, such as "2017-06" for "back in 2017", is unsupported.
- "contradicted": the transcript says something else.
A claim marked disputed is "supported" when its teller said it, even if someone else disagreed. A tie between two members of a named group is "supported" when the transcript lists both in it. Choose the verdict that your note gives. Names in the transcript come from speech recognition and may be misspelled. Mark "trivia" true when the item is small talk, a joke, a question, or a passing remark that is not lasting history about the people. Write the note first, in at most 15 words."""

RUBRIC_SYSTEM = f"""You grade the record that software wrote about a conversation among friends. You get a gold checklist of what a perfect record captures, the record, and the findings of checks that compared each gold item and each record item one by one. Treat the findings as settled: call a gold item missing only when the findings list it, and base each score on them. Then judge what the checks cannot see: how specific the labels are, whether stories carry the complication, the outcome, and vivid details, and whether a friend could retell the group's history from the record. Score each dimension from 1 to 5 with a reason of at most 25 words that names findings or record items. Write the reason before the score.

{RUBRIC}"""

PAIR_SYSTEM = f"""You compare two records that two runs of software wrote about the same conversation among friends. You get the transcript, a gold checklist of what a perfect record captures, record A, and record B. For each dimension, write a reason of at most 25 words, then say which record is better: "A", "B", or "tie" when they are about equal. A wrong fact counts against a record more than a missing one.

{RUBRIC}"""

CREDIT = {"yes": 1.0, "partly": 0.5, "no": 0.0, "supported": 1.0, "unsupported": 0.0, "contradicted": 0.0}


@dataclass(frozen=True)
class Run:
    """The text of a run that the prompts show, and the items that precision checks."""

    listing: str
    transcript: str
    precision_items: list[Record]


def describe(graph: Graph, recorded_at: str) -> Run:
    """Numbers the items of each section in the order the run wrote them, so an unchanged replay gives the same
    text."""

    def name(person_id: int | None) -> str:
        return graph.people[person_id]["name"] if person_id in graph.people else "unknown"

    room = set(graph.room())
    lines = ["PEOPLE:"]
    lines += [
        f"p{i} {p['name']} ({'in the room' if pid in room else 'mentioned'})"
        for i, (pid, p) in enumerate(graph.people.items(), 1)
    ]
    relationships = [
        {
            "id": f"r{i}",
            "text": f'{name(r["a"])} - {name(r["b"])}: {r["category"]} "{r["label"]}" {r["status"]}'
            + (f", met {r['met_on']}" if r["met_on"] else "")
            + (f". {r['summary']}" if r["summary"] else ""),
        }
        for i, r in enumerate(graph.relationships.values(), 1)
    ]
    lines += ["", "RELATIONSHIPS:"] + [f"{r['id']} {r['text']}" for r in relationships]
    lines += ["", "EVENTS:"]
    lines += [
        f"e{i} {e['name']} ({e['start_date'] or e['when_text'] or 'no date'}"
        + (f", {e['place']}" if e["place"] else "")
        + f"; {', '.join(name(p['person_id']) for p in e['people'])})"
        for i, e in enumerate(graph.events.values(), 1)
    ]
    live = graph.live_facts()
    facts = []
    for i, f in enumerate((f for f in live if f["segment_ids"]), 1):
        flags = [f["kind"], f["start_date"] or "no date"]
        if f["told_by_person_id"]:
            flags.append(f"told by {name(f['told_by_person_id'])}")
        if f["disputed"]:
            flags.append("disputed")
        facts.append({"id": f"f{i}", "text": f"[{', '.join(flags)}] {f['text']}"})
    lines += ["", "FACTS:"] + [f"{f['id']} {f['text']}" for f in facts]
    lines += ["", "STORIES:"]
    lines += [
        f"s{i} {f['text']} ({f['start_date'] or 'no date'}; {', '.join(name(p['person_id']) for p in f['people'])}): "
        f"{f['detail'] or ''}"
        for i, f in enumerate((f for f in live if not f["segment_ids"]), 1)
    ]
    lines += ["", "GROUPS:"]
    lines += [
        f"g{i} {g['name']}: "
        + ", ".join(name(m["person_id"]) for m in graph.members.values() if m["group_id"] == g["id"])
        for i, g in enumerate(graph.groups.values(), 1)
    ]
    transcript = [f"Session date: {recorded_at}"]
    transcript += [
        f"{name(s['person_id']) if s['person_id'] else 'unknown'}: {s['text']}"
        for s in sorted(graph.segments.values(), key=lambda s: s["start_ms"])
    ]
    return Run("\n".join(lines), "\n".join(transcript), relationships + facts)


def gold_sections(gold: Record) -> dict[str, list[Record]]:
    """Every list section of the checklist is a recall category, so a new section needs no code change.
    `not_facts` holds plain strings that a good run leaves out."""
    sections = {key: value for key, value in gold.items() if isinstance(value, list) and value}
    if "not_facts" in sections:
        sections["not_facts"] = [{"id": f"n{i}", "text": text} for i, text in enumerate(sections["not_facts"], 1)]
    return sections


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def list_schema(ids: list[str], properties: dict) -> dict:
    item = object_schema({"id": {"type": "string", "enum": ids}, **properties})
    return object_schema({"items": {"type": "array", "items": item, "minItems": len(ids), "maxItems": len(ids)}})


def recall_call(category: str, items: list[Record], run: Run, *, model: str, seed: int) -> Call:
    schema = list_schema(
        [item["id"] for item in items],
        {
            "note": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
        },
    )
    user = (
        f"RECORD:\n{run.listing}\n\nGOLD {category.upper()}:\n"
        + "\n".join(json.dumps(item, ensure_ascii=False) for item in items)
        + "\n\nAnswer once for each gold item, in order."
    )
    return Call("judge", model, PROMPT_VERSION, RECALL_SYSTEM, user, schema, seed, max_tokens=4000)


def precision_call(items: list[Record], run: Run, *, model: str, seed: int) -> Call:
    schema = list_schema(
        [item["id"] for item in items],
        {
            "note": {"type": "string"},
            "support": {"type": "string", "enum": ["supported", "partly", "unsupported", "contradicted"]},
            "trivia": {"type": "boolean"},
        },
    )
    user = (
        f"TRANSCRIPT:\n{run.transcript}\n\nRECORD ITEMS:\n"
        + "\n".join(f"{item['id']} {item['text']}" for item in items)
        + "\n\nAnswer once for each record item, in order."
    )
    return Call("judge", model, PROMPT_VERSION, PRECISION_SYSTEM, user, schema, seed, max_tokens=4000)


def rubric_call(gold: Record, run: Run, findings: str, *, model: str, seed: int) -> Call:
    dimension = object_schema({"reason": {"type": "string"}, "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}})
    schema = object_schema({name: dimension for name in DIMENSIONS})
    user = (
        f"GOLD CHECKLIST:\n{json.dumps(gold, ensure_ascii=False)}\n\nRECORD TO GRADE:\n{run.listing}\n\n"
        f"FINDINGS OF THE ITEM CHECKS:\n{findings}"
    )
    return Call("judge", model, RUBRIC_VERSION, RUBRIC_SYSTEM, user, schema, seed, max_tokens=2000)


def findings(result: Record) -> str:
    """The gold items that the record misses or only partly captures, and the record items that the transcript
    does not support, so the rubric grades from the same verdicts as recall and precision."""

    def gold_line(key: str, item: Record) -> str:
        body = json.dumps({k: v for k, v in item.items() if k not in ("id", "judge")}, ensure_ascii=False)
        return f"- {key} {item['id']}: {body[:200]} ({item['judge'].get('note', '') if item['judge'] else 'no answer'})"

    def verdict(item: Record) -> str | None:
        return (item["judge"] or {}).get("verdict", "no")

    sections = {key: items for key, items in result["items"].items() if key not in ("precision", "not_facts")}
    lines = ["Gold items the record misses:"]
    lines += [gold_line(k, i) for k, items in sections.items() for i in items if verdict(i) == "no"] or ["- none"]
    lines += ["Gold items the record only partly captures:"]
    lines += [gold_line(k, i) for k, items in sections.items() for i in items if verdict(i) == "partly"] or ["- none"]
    lines += ["Small talk the record states as a fact:"]
    lines += [f"- {i['text']}" for i in result["items"].get("not_facts", []) if verdict(i) == "yes"] or ["- none"]
    lines += ["Record items the transcript does not fully support, or that are trivia:"]
    lines += [
        f"- {i['id']} {i['text'][:160]}: {(i['judge'] or {}).get('support', 'unsupported')}"
        + (", trivia" if (i["judge"] or {}).get("trivia") else "")
        for i in result["items"]["precision"]
        if not i["judge"] or i["judge"]["support"] != "supported" or i["judge"].get("trivia")
    ] or ["- none"]
    return "\n".join(lines)


def pair_call(gold: Record, first: Run, second: Run, *, model: str, seed: int) -> Call:
    dimension = object_schema({"reason": {"type": "string"}, "better": {"type": "string", "enum": ["A", "B", "tie"]}})
    schema = object_schema({name: dimension for name in DIMENSIONS})
    user = (
        f"TRANSCRIPT:\n{first.transcript}\n\nGOLD CHECKLIST:\n{json.dumps(gold, ensure_ascii=False)}\n\n"
        f"RECORD A:\n{first.listing}\n\nRECORD B:\n{second.listing}"
    )
    return Call("judge", model, PROMPT_VERSION, PAIR_SYSTEM, user, schema, seed, max_tokens=2000)


def answers(output: Record | None, ids: list[str]) -> dict[str, Record]:
    """The first answer for each asked id. An id the model skipped, or a call that returned no JSON, gets no
    answer and scores as a miss."""
    found: dict[str, Record] = {}
    for answer in (output or {}).get("items", []):
        found.setdefault(answer["id"], answer)
    return {i: found[i] for i in ids if i in found}


def recall(items: list[Record], found: dict[str, Record]) -> float:
    return sum(CREDIT[found[i["id"]]["verdict"]] for i in items if i["id"] in found) / len(items)


def precision(items: list[Record], found: dict[str, Record]) -> Record:
    supports = [found[i["id"]]["support"] if i["id"] in found else "unsupported" for i in items]
    return {
        "precision": sum(CREDIT[s] for s in supports) / len(items) if items else None,
        "contradicted": supports.count("contradicted"),
        "unsupported": supports.count("unsupported"),
        "trivia": sum(1 for i in items if found.get(i["id"], {}).get("trivia")),
    }


def agree(forward: Record | None, backward: Record | None) -> dict[str, str]:
    """Keeps a verdict only when the same run wins in both orders. `backward` saw the runs swapped, so its "A" is
    run B. A disagreement, or a call without JSON, is a tie."""
    swapped = {"A": "B", "B": "A", "tie": "tie"}
    verdicts = {}
    for name in DIMENSIONS:
        first = (forward or {}).get(name, {}).get("better")
        second = swapped.get((backward or {}).get(name, {}).get("better"))
        verdicts[name] = first if first is not None and first == second else "tie"
    return verdicts


async def gather(llm: LLM, calls: list[Call], *, run_id: int) -> list[Record | None]:
    limit = asyncio.Semaphore(PARALLEL)

    async def one(call: Call) -> Record | None:
        async with limit:
            return (await llm.call(call, run_id=run_id, audio_ms=0)).output

    return await asyncio.gather(*(one(call) for call in calls))


async def judge_run(llm: LLM, gold: Record, run: Run, *, run_id: int, model: str, seed: int) -> Record:
    sections = gold_sections(gold)
    items = run.precision_items
    batches = [items[i : i + PRECISION_BATCH] for i in range(0, len(items), PRECISION_BATCH)]
    calls = [recall_call(key, section, run, model=model, seed=seed) for key, section in sections.items()]
    calls += [precision_call(batch, run, model=model, seed=seed) for batch in batches]
    outputs = await gather(llm, calls, run_id=run_id)
    recall_outputs, precision_outputs = outputs[: len(sections)], outputs[len(sections) :]

    result: Record = {"run_id": run_id, "sample": gold.get("sample"), "seed": seed, "recall": {}, "items": {}}
    for (key, section), output in zip(sections.items(), recall_outputs, strict=True):
        found = answers(output, [i["id"] for i in section])
        result["recall"][key] = recall(section, found)
        result["items"][key] = [{**item, "judge": found.get(item["id"])} for item in section]
    scored = [key for key in sections if key != "not_facts"]
    total = sum(len(sections[key]) for key in scored)
    result["recall_all"] = sum(result["recall"][key] * len(sections[key]) for key in scored) / max(1, total)

    found = {}
    for batch, output in zip(batches, precision_outputs, strict=True):
        found |= answers(output, [i["id"] for i in batch])
    result["precision"] = {
        "facts": precision([i for i in items if i["id"].startswith("f")], found),
        "relationships": precision([i for i in items if i["id"].startswith("r")], found),
        "all": precision(items, found),
    }
    result["items"]["precision"] = [{**item, "judge": found.get(item["id"])} for item in items]

    [rubric] = await gather(llm, [rubric_call(gold, run, findings(result), model=model, seed=seed)], run_id=run_id)
    result["rubric"] = {name: (rubric or {}).get(name) for name in DIMENSIONS}
    scores = [d["score"] for d in result["rubric"].values() if d]
    result["rubric_mean"] = sum(scores) / len(scores) if scores else None
    return result


async def judge_pair(llm: LLM, gold: Record, a: Run, b: Run, *, run_id: int, model: str, seed: int) -> Record:
    calls = [pair_call(gold, a, b, model=model, seed=seed), pair_call(gold, b, a, model=model, seed=seed)]
    forward, backward = await gather(llm, calls, run_id=run_id)
    return {"verdicts": agree(forward, backward), "forward": forward, "backward": backward}
