"""Turns one speaker turn into people, facts, events, groups, and relationships."""

import re
from dataclasses import dataclass, field

from grapevine.config import Context
from grapevine.graph import Record
from grapevine.llm import Call
from grapevine.prompter import show_prompt
from grapevine.store import Store, name_matches

PROMPT_VERSION = "extract-v3"

SYSTEM = """You extract facts about people from one turn of a live group conversation. The people in the room introduced themselves at the start. Return JSON only.

The input lists today's date, the known people, events (ids like e2), groups, relationships, recent facts (ids like f7), the recent turns, and the NEW TURN with its speaker.

PEOPLE
- Refer to a person by the full name in KNOWN PEOPLE, for example "Ray Patel".
- "speaker" is the person who said the new turn. Map "I", "me", and "my" to "speaker". Map "we" to "speaker" plus the people the speaker means.
- For a new person, write the name as said, for example "Priya", and add them to "people".
- If a first name fits more than one known person and the turn does not say which one, write only that first name, for example "Andrew".
- List in "people" only new people, or known people with a new alias or gender.
- gender: male or female only with a clear signal (a pronoun, a gendered word, a clearly gendered name). Else unknown.

SPEAKER NAME
- If the speaker is unknown and says their own name, set "speaker_name" to the full name they say. Else null.

FACTS
- Write one fact for each new piece of information in the NEW TURN only. Do not repeat known facts. Skip questions and filler.
- kind: personal (one person: job, home, school, pet, hobby), relationship (how two people know each other), story (something that happened), membership, correction.
- If the turn names a group of people, such as a crew, a team, a band, a club, or a friend group, write a membership fact with "group" set to the group name and "people" set to its members. A group is not an event.
- text: one short sentence with full names and no pronouns, for example "Ray Patel and Jasper Lin met freshman year at Cornell." State the fact itself, not that someone said or corrected it.
- people: every person the fact is about, by name.
- when_text: the time as said, for example "freshman year". start_date and end_date: "YYYY", "YYYY-MM", or "YYYY-MM-DD" when the turn, the known facts, or today's date resolve it. Else null.
- event: for a named or dated happening (a festival, a trip, a wedding, a proposal, an accident), the event id ("e2") or the new event name, also listed in "events". Reuse a known event when the turn talks about it again. Name an event by what and when, for example "Electric Zoo 2018". All facts of one story share one event. A school, a place, or a group is not an event.
- If the speaker corrects a fact that they told, or agrees with a correction, use kind correction and set "revises" to the old fact id.
- If the speaker contradicts a fact that ANOTHER person told, write the speaker's version and set "disputes" to the other fact id.

RELATIONSHIPS
- Add one entry for each pair whose relationship the turn tells about, for example coworkers, cousins, roommates, exes, or old friends. A pair has at most one entry per category, for example a past romantic one and a current friend one.
- category: romantic, family, friend, professional, or other.
- label: a 1 to 3 word term, for example "college roommate", "ex-girlfriend", "cousin", "coworker".
- status: current or past. met_on: when they met as a date, or null. met_event: an event id or name, or null.
- closeness 1 to 5, recency 1 to 5 (5 means in touch now), sentiment -2 to 2. Null when the turn gives no signal.
- summary: one sentence on how they relate.
- facts: the 0-based indexes of the facts in your output that support it.

Return empty lists when the turn has no new information.

EXAMPLE INPUT
KNOWN PEOPLE:
- Maria Lopez (aliases: Maria) [in the room]
- Tom Becker (aliases: Tom) [in the room]
- Andrew Kim (aliases: Andrew)
- Andrew Shaw (aliases: Andrew)
FACTS:
f4 told by Tom Becker: Tom Becker and Maria Lopez met in 2015.
NEW TURN (speaker: Maria Lopez):
No, we met in 2014 at Lollapalooza. Andrew drove us. My sister Ana was there too, she runs the Saturday book club with me.

EXAMPLE OUTPUT
{"speaker_name": null, "people": [{"name": "Ana", "aliases": [], "gender": "female"}], "events": [{"ref": "Lollapalooza 2014", "name": "Lollapalooza 2014", "when_text": "2014", "start_date": "2014", "end_date": null, "place": null, "people": ["speaker", "Tom Becker", "Ana"]}], "facts": [{"kind": "relationship", "text": "Maria Lopez and Tom Becker met at Lollapalooza in 2014.", "people": ["speaker", "Tom Becker"], "when_text": "2014", "start_date": "2014", "end_date": null, "event": "Lollapalooza 2014", "group": null, "revises": null, "disputes": "f4"}, {"kind": "story", "text": "Andrew drove Maria Lopez and Tom Becker to Lollapalooza.", "people": ["Andrew", "speaker", "Tom Becker"], "when_text": "2014", "start_date": "2014", "end_date": null, "event": "Lollapalooza 2014", "group": null, "revises": null, "disputes": null}, {"kind": "membership", "text": "Ana and Maria Lopez run the Saturday book club.", "people": ["Ana", "speaker"], "when_text": null, "start_date": null, "end_date": null, "event": null, "group": "Saturday book club", "revises": null, "disputes": null}], "relationships": [{"a": "speaker", "b": "Tom Becker", "category": "friend", "label": "friend", "status": "current", "met_on": "2014", "met_event": "Lollapalooza 2014", "closeness": null, "recency": null, "sentiment": null, "summary": "Maria Lopez and Tom Becker met at Lollapalooza in 2014.", "facts": [0]}, {"a": "speaker", "b": "Ana", "category": "family", "label": "sister", "status": "current", "met_on": null, "met_event": null, "closeness": 4, "recency": 5, "sentiment": 1, "summary": "Ana is Maria Lopez's sister.", "facts": [2]}]}"""

NULLABLE = {"type": ["string", "null"]}
DATE = {"type": ["string", "null"], "description": "YYYY, YYYY-MM, or YYYY-MM-DD"}
REFS = {"type": "array", "items": {"type": "string"}}
SCORE = {"type": ["integer", "null"]}


def _object(**properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


SCHEMA = _object(
    speaker_name=NULLABLE,
    people={
        "type": "array",
        "items": _object(
            name={"type": "string"},
            aliases=REFS,
            gender={"type": "string", "enum": ["male", "female", "unknown"]},
        ),
    },
    events={
        "type": "array",
        "items": _object(
            ref={"type": "string"},
            name={"type": "string"},
            when_text=NULLABLE,
            start_date=DATE,
            end_date=DATE,
            place=NULLABLE,
            people=REFS,
        ),
    },
    facts={
        "type": "array",
        "items": _object(
            kind={"type": "string", "enum": ["personal", "relationship", "story", "membership", "correction"]},
            text={"type": "string"},
            people=REFS,
            when_text=NULLABLE,
            start_date=DATE,
            end_date=DATE,
            event=NULLABLE,
            group=NULLABLE,
            revises=NULLABLE,
            disputes=NULLABLE,
        ),
    },
    relationships={
        "type": "array",
        "items": _object(
            a={"type": "string"},
            b={"type": "string"},
            category={"type": "string", "enum": ["romantic", "family", "friend", "professional", "other"]},
            label={"type": "string"},
            status={"type": "string", "enum": ["current", "past"]},
            met_on=DATE,
            met_event=NULLABLE,
            closeness=SCORE,
            recency=SCORE,
            sentiment=SCORE,
            summary={"type": "string"},
            facts={"type": "array", "items": {"type": "integer"}},
        ),
    },
)

DATE_PATTERN = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")
SEASONS = ("spring", "summer", "fall", "autumn", "winter")
SPEAKER_REFS = {"speaker", "the speaker", "unknown speaker", "i", "me", "myself"}
RECENT_TURNS = 4


@dataclass(frozen=True)
class Teller:
    """Who said a turn, and which turn."""

    segment_id: int
    person_id: int | None
    confidence: float
    model_call_id: int


@dataclass(frozen=True)
class Ambiguity:
    mention: str
    candidates: list[int]


@dataclass
class Held:
    """Items of one model output that wait for an answer, with what is already resolved."""

    output: Record
    teller: Teller
    answers: dict[str, int] = field(default_factory=dict)
    fact_ids: dict[int, int] = field(default_factory=dict)
    done_relationships: list[int] = field(default_factory=list)


def clean_date(value: str | None) -> str | None:
    return value if value and DATE_PATTERN.fullmatch(value) else None


def year_in(text: str) -> str | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match[0] if match else None


def date_precision(date: str | None, when_text: str | None) -> str | None:
    if date is None:
        return None
    if when_text and len(date) < 10 and any(s in when_text.lower() for s in SEASONS):
        return "season"
    return {4: "year", 7: "month", 10: "day"}[len(date)]


def person_line(person: Record, room: set[int]) -> str:
    aliases = f" (aliases: {', '.join(person['aliases'])})" if person["aliases"] else ""
    return f"- {person['name']}{aliases}{' [in the room]' if person['id'] in room else ''}"


def mentioned(text: str, person: Record) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    names = [person["name"], *person["aliases"]]
    return any(set(n.lower().split()) <= words for n in names) or person["name"].lower().split()[0] in words


def user_message(store: Store, *, segment: Record, speaker_id: int | None, today: str, context: Context) -> str:
    """Lists only what this run already knows. People from other sessions still match by name when the
    output is applied, so an unchanged replay sends the same input."""
    graph = store.graph
    room = {pid for pid in graph.speakers.values() if pid is not None}
    turns = sorted(graph.segments.values(), key=lambda s: s["start_ms"])
    recent = [s for s in turns if s["start_ms"] < segment["start_ms"]][-RECENT_TURNS:]
    window = " ".join(s["text"] for s in [*recent, segment])

    everyone = {p["id"]: p for p in graph.people.values() if p["merged_into_id"] is None}
    if context == "relevant":
        focus = room | {pid for pid, p in everyone.items() if mentioned(window, p)}
    else:
        focus = set(everyone)
    people = [p for pid, p in everyone.items() if pid in focus or pid in room]

    def touches(ids: list[int]) -> bool:
        return context == "full" or bool(set(ids) & focus)

    def name(person_id: int | None) -> str:
        return everyone[person_id]["name"] if person_id in everyone else "unknown speaker"

    lines = [f"TODAY: {today}", "", "KNOWN PEOPLE:"]
    lines += [person_line(p, room) for p in people] or ["(none)"]
    lines += ["", "EVENTS:"]
    lines += [
        f"{graph.ref('e', e['id'])} {e['name']} ({e['start_date'] or e['when_text'] or 'no date'})"
        for e in graph.events.values()
        if touches([p["person_id"] for p in e["people"]])
    ] or ["(none)"]
    lines += ["", "GROUPS: " + (", ".join(g["name"] for g in graph.groups.values()) or "(none)"), "", "RELATIONSHIPS:"]
    lines += [
        f'{name(r["a"])} - {name(r["b"])}: {r["category"]} "{r["label"]}" {r["status"]}'
        + (f", met {r['met_on']}" if r["met_on"] else "")
        for r in graph.relationships.values()
        if touches([r["a"], r["b"]])
    ] or ["(none)"]
    lines += ["", "FACTS:"]
    lines += [
        f"{graph.ref('f', f['id'])} told by {name(f['told_by_person_id'])}: {f['text']}"
        + (f" [{f['start_date']}]" if f["start_date"] else "")
        + (" (disputed)" if f["disputed"] else "")
        for f in graph.live_facts()
        if touches([p["person_id"] for p in f["people"]])
    ] or ["(none)"]
    lines += ["", "RECENT TURNS:"]
    lines += [f"{name(s['person_id'])}: {s['text']}" for s in recent] or ["(none)"]
    speaker = name(speaker_id) if speaker_id else f"unknown, voice {segment['speaker_label']}"
    lines += ["", f"NEW TURN (speaker: {speaker}):", segment["text"]]
    return "\n".join(lines)


def extraction_call(
    store: Store, *, segment: Record, speaker_id: int | None, today: str, model: str, seed: int, context: Context
) -> Call:
    user = user_message(store, segment=segment, speaker_id=speaker_id, today=today, context=context)
    return Call("extractor", model, PROMPT_VERSION, SYSTEM, user, SCHEMA, seed)


def apply_output(store: Store, held: Held) -> None:
    """Writes a model output. Items that name an ambiguous person wait in a prompt until someone answers."""
    output, teller = held.output, held.teller
    new_people = {p["name"].lower(): p for p in output["people"]}
    text = store.graph.segments[teller.segment_id]["text"] if teller.segment_id in store.graph.segments else ""
    resolved: dict[str, int | None | Ambiguity] = {}

    def person(ref: str) -> int | None | Ambiguity:
        if ref not in resolved:
            resolved[ref] = resolve(
                store, ref, new_people.get(ref.lower()), teller=teller, answers=held.answers, text=text
            )
        return resolved[ref]

    def people(refs: list[str]) -> list[int] | Ambiguity:
        ids = []
        for ref in refs:
            match person(ref):
                case Ambiguity() as ambiguity:
                    return ambiguity
                case None:
                    continue
                case person_id:
                    ids.append(person_id)
        return list(dict.fromkeys(ids))

    waiting: dict[str, Ambiguity] = {}
    with store.batch():
        events = apply_events(store, output["events"], people, teller)

        for index, fact in enumerate(output["facts"]):
            if index in held.fact_ids:
                continue
            ids = people(fact["people"])
            if isinstance(ids, Ambiguity):
                waiting[ids.mention] = ids
                continue
            held.fact_ids[index] = apply_fact(store, fact, ids, events, teller)

        for index, rel in enumerate(output["relationships"]):
            if index in held.done_relationships:
                continue
            ids = people([rel["a"], rel["b"]])
            if isinstance(ids, Ambiguity):
                waiting[ids.mention] = ids
                continue
            held.done_relationships.append(index)
            if len(ids) != 2:
                continue
            evidence = [held.fact_ids[i] for i in rel["facts"] if i in held.fact_ids]
            store.upsert_relationship(
                a=ids[0],
                b=ids[1],
                category=rel["category"],
                fields={
                    "label": rel["label"],
                    "summary": rel["summary"],
                    "status": rel["status"],
                    "met_on": clean_date(rel["met_on"]) or met_on(store, evidence),
                    "met_event_id": events.get(rel["met_event"] or ""),
                    "closeness": rel["closeness"],
                    "recency": rel["recency"],
                    "sentiment": rel["sentiment"],
                },
                evidence=evidence,
                model_call_id=teller.model_call_id,
            )

        for ambiguity in waiting.values():
            names = [store.person(c)["name"] for c in ambiguity.candidates]
            show_prompt(
                store,
                text=f"Which {ambiguity.mention}: {' or '.join(names)}?",
                reason="which_person",
                person_ids=ambiguity.candidates,
                payload={"mention": ambiguity.mention, "held": held_payload(held)},
                model_call_id=teller.model_call_id,
            )


def met_on(store: Store, evidence: list[int]) -> str | None:
    """The date of a supporting fact that tells how the pair met."""
    for fact_id in evidence:
        fact = store.graph.facts[fact_id]
        if fact["start_date"] and re.search(r"\bmet\b|\bmeet\b", fact["text"].lower()):
            return fact["start_date"]
    return None


def resolve(
    store: Store, ref: str, entry: Record | None, *, teller: Teller, answers: dict[str, int], text: str
) -> int | None | Ambiguity:
    """Finds the person a model ref names, creates a new person, or reports that the name fits several people."""
    if ref.strip().lower() in SPEAKER_REFS:
        return teller.person_id
    if ref in answers:
        return answers[ref]
    matches = name_matches(store.conn, ref)
    if not matches and entry:
        matches = name_matches(store.conn, entry["name"])
    if len(matches) > 1:
        return Ambiguity(ref, [p["id"] for p in matches])
    if not matches:
        name = entry["name"] if entry else ref
        aliases = entry["aliases"] if entry else []
        if " " in name and name.split()[0] not in aliases:
            aliases = [name.split()[0], *aliases]
        return store.create_person(name=name, aliases=aliases, gender=entry["gender"] if entry else "unknown")["id"]

    person = matches[0]
    if ambiguity := first_name_only(store, person, text):
        return answers.get(ambiguity.mention, ambiguity)
    if entry:
        store.update_person(person["id"], add_aliases=entry["aliases"], gender=entry["gender"])
    return person["id"]


def first_name_only(store: Store, person: Record, text: str) -> Ambiguity | None:
    """The model may pick one Andrew when the turn only says "Andrew". The room decides instead."""
    first = person["name"].split()[0]
    others = name_matches(store.conn, first)
    if len(others) < 2:
        return None
    said = set(re.findall(r"[a-z']+", text.lower()))
    shared = {first.lower()}
    distinct = [n for n in [person["name"], *person["aliases"]] if n.lower() not in shared]
    if first.lower() not in said or any(set(n.lower().split()) <= said for n in distinct):
        return None
    return Ambiguity(first, [p["id"] for p in others])


def apply_events(store: Store, events: list[Record], people, teller: Teller) -> dict[str, int]:
    """Writes the events of an output and maps each event ref to its id."""
    ids: dict[str, int] = {}
    for event in events:
        event_id = store.graph.lookup("e", event["ref"])
        if event_id is None:
            event_id = next(
                (e["id"] for e in store.graph.events.values() if e["name"].lower() == event["name"].lower()), None
            )
        attendees = people(event["people"])
        start = clean_date(event["start_date"]) or year_in(event["name"])
        record = store.upsert_event(
            event_id,
            {
                "name": event["name"],
                "when_text": event["when_text"],
                "start_date": start,
                "end_date": clean_date(event["end_date"]),
                "date_precision": date_precision(start, event["when_text"]),
                "place": event["place"],
            },
            people=attendees if isinstance(attendees, list) else [],
            model_call_id=teller.model_call_id,
        )
        ids[event["ref"]] = ids[event["name"]] = record["id"]
    for event_id in store.graph.events:
        ids.setdefault(store.graph.ref("e", event_id), event_id)
    return ids


def apply_fact(store: Store, fact: Record, people: list[int], events: dict[str, int], teller: Teller) -> int:
    start = clean_date(fact["start_date"])
    event_id = events.get(fact["event"] or "")
    roles = [(pid, "subject") for pid in people]
    if teller.person_id is not None and teller.person_id not in people:
        roles.append((teller.person_id, "teller"))
    record = store.add_fact(
        {
            "kind": fact["kind"],
            "text": fact["text"],
            "told_by_person_id": teller.person_id,
            "when_text": fact["when_text"],
            "start_date": start,
            "end_date": clean_date(fact["end_date"]),
            "date_precision": date_precision(start, fact["when_text"]),
            "event_id": event_id,
            "confidence": teller.confidence,
        },
        people=roles,
        segment_ids=[teller.segment_id],
        model_call_id=teller.model_call_id,
    )
    if event_id is not None and people:
        store.upsert_event(event_id, {}, people=people, model_call_id=teller.model_call_id)

    if fact["kind"] == "membership" and fact["group"]:
        group = store.group_for(fact["group"])
        for person_id in people:
            store.add_member(
                group_id=group["id"],
                person_id=person_id,
                since=start,
                until=clean_date(fact["end_date"]),
                fact_id=record["id"],
            )

    old = store.graph.lookup("f", fact["revises"]) or store.graph.lookup("f", fact["disputes"])
    if old is None:
        return record["id"]
    told_by = store.graph.facts[old]["told_by_person_id"]
    if told_by is None or told_by == teller.person_id:
        revise(store, old, record["id"], model_call_id=teller.model_call_id)
    else:
        dispute(store, old, record["id"], model_call_id=teller.model_call_id)
    return record["id"]


def revise(store: Store, old: int, new: int, *, model_call_id: int) -> None:
    """A teller corrects their own fact, or agrees with a correction. That settles a dispute about the fact."""
    store.update_fact(old, revised_by_fact_id=new, model_call_id=model_call_id)
    for prompt in store.graph.active_prompts():
        if prompt["reason"] == "conflict" and old in prompt["payload"]["fact_ids"]:
            store.dismiss_prompt(prompt["id"])
            for fact_id in prompt["payload"]["fact_ids"]:
                if fact_id != old and store.graph.facts[fact_id]["disputed"]:
                    store.update_fact(fact_id, disputed=0, model_call_id=model_call_id)


def dispute(store: Store, old: int, new: int, *, model_call_id: int) -> None:
    """Two people disagree. Both facts stay, and the room is asked once per dispute."""
    already = store.graph.facts[old]["disputed"]
    store.update_fact(new, disputed=1, model_call_id=model_call_id)
    if already:
        return
    store.update_fact(old, disputed=1, model_call_id=model_call_id)
    first, second = store.graph.facts[old], store.graph.facts[new]
    tellers = [
        store.person(f["told_by_person_id"])["name"].split()[0] for f in (first, second) if f["told_by_person_id"]
    ]
    show_prompt(
        store,
        text=f"{' and '.join(tellers) if len(tellers) == 2 else 'People'} disagree. “{first['text']}” or “{second['text']}”?",
        reason="conflict",
        person_ids=[f["told_by_person_id"] for f in (first, second) if f["told_by_person_id"]],
        payload={"fact_ids": [old, new]},
        model_call_id=model_call_id,
    )


def held_payload(held: Held) -> Record:
    return {
        "output": held.output,
        "teller": held.teller.__dict__,
        "answers": held.answers,
        "fact_ids": {str(k): v for k, v in held.fact_ids.items()},
        "done_relationships": held.done_relationships,
    }


def held_from_payload(payload: Record) -> Held:
    return Held(
        output=payload["output"],
        teller=Teller(**payload["teller"]),
        answers=payload["answers"],
        fact_ids={int(k): v for k, v in payload["fact_ids"].items()},
        done_relationships=payload["done_relationships"],
    )
