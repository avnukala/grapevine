"""Turns one speaker turn into people, facts, events, groups, and relationships."""

import re
from dataclasses import dataclass, field

from grapevine.config import Context
from grapevine.graph import Record
from grapevine.llm import LLM, Call, watch_on
from grapevine.names import better_spelling, drop_ly, misspelled, near_first, pairs_of
from grapevine.prompter import show_prompt
from grapevine.store import Store, name_matches, proper_group_name

PROMPT_VERSION = "extract-v15"

SYSTEM = """You extract facts about people from one turn of a live group conversation. The people in the room introduced themselves at the start. Return JSON only.

Every word you write delays the live view, so leave out any optional field that would be null, empty, or false.

The input lists today's date, the known people, events (ids like e2), groups, relationships, recent facts (ids like f7), the recent turns, and the NEW TURN with its speaker.

PEOPLE
- Refer to a person by the full name in KNOWN PEOPLE, for example "Marcus Hale".
- "speaker" is the person who said the new turn. Map "I", "me", and "my" to "speaker". Map "we" to "speaker" plus the people the speaker means. "You" is the person the speaker talks to, and "you two" is that person and the one the recent turns pair them with, such as a partner just named, never someone else in the room.
- For a new person, write the name as said, for example "Imani", and add them to "people".
- If a first name fits more than one known person and the turn does not say which one, write only that first name, for example "Felix".
- List in "people" only new people, or known people with a new alias or gender.
- gender: male or female only with a clear signal (a pronoun, a gendered word, a clearly gendered name). Else unknown.

SPEAKER NAME
- If the speaker is unknown and says their own name, set "speaker_name" to the full name they say. Else null.

FACTS
- Write one fact for each new piece of information in the NEW TURN only. Do not repeat known facts. Skip questions and filler.
- Speech recognition mishears words. Skip a line that makes no sense in the talk, such as a greeting or a toast that reads like a claim.
- A short turn such as "Lisbon, 2019." tells about the topic of the turn right before it, not about an older fact.
- kind: personal (one person: job, home, school, pet, hobby), relationship (how two people know each other, such as "Dana Cole is Mia Rao's manager" or "Leo Hale is Tess's ex"), story (something that happened), membership, correction.
- If the turn names a group of people, such as a crew, a team, a band, a club, or a friend group, write a membership fact with "group" set to the group name and "people" set to its members. When people in the room say "we" or "our team", they are members. A person who leads or runs the group, such as its commissioner, captain, or president, is a member. If the turn calls a known group something else, such as "the crew", use the known group name. A group is not an event, and a house is not a group.
- text: one short sentence with full names and no pronouns, for example "Marcus Hale and Tina Alvarez met freshman year at Rutgers." State the fact itself, not that someone said or corrected it.
- people: every person the fact is about, by name.
- when_text: the time as said, for example "freshman year". start_date and end_date: "YYYY", "YYYY-MM", or "YYYY-MM-DD" when the turn, the known facts, or today's date resolve it. "Tonight" and "today" are today's date. A past happening with no time said gets no date, never today's date.
- event: for a named or dated happening (a festival, a show or concert, including one tonight, a trip, a wedding, a proposal, an accident, a move to a new city or house), the event name, also listed in "events". To talk about a known event again, list it in "events" with its "id" ("e2") and use its known name. Name a new event by what and when, for example "Coachella 2018". All facts of one story share one event. A school, a place, or a group is not an event.
- If the speaker corrects a fact that they told, or agrees with a correction, use kind correction and set "revises" to the old fact id. A speaker who gives in, such as with "okay, fine" or "you are right", agrees with the correction. Write that correction first, even though it repeats a known fact. For example, Mia Rao told f3 "Mia Rao moved to Seattle in 2016.", Dana Cole said it was 2017, and Mia Rao now says "Okay, fine, it was 2017, and I love it there": the first fact is {"kind": "correction", "text": "Mia Rao moved to Seattle in 2017.", "people": ["speaker"], "start_date": "2017", "revises": "f3"}.
- If the speaker contradicts a fact that ANOTHER person told, write the speaker's version and set "disputes" to the other fact id. Set "disputes" only when the two facts give different values for the same thing, such as two years for one trip. Added detail, an excuse, or an answer to a question does not dispute, for example "It was raining!" after "Felix crashed the car".
- A turn that starts with "that is true, but" or "yes, and" agrees with the turn before. Its new facts do not dispute anything.

RELATIONSHIPS
- Add one entry for each pair whose relationship the turn tells about, for example coworkers, cousins, roommates, exes, or old friends. A pair has at most one entry per category, for example a past romantic one and a current friend one.
- Tie the pair the turn names. "We met at my cousin's wedding" makes the speaker a cousin of the one getting married, not of the person they met.
- Every fact of kind relationship needs a matching entry here. "My coworker Wren", "my manager Dana", and "I'm the ex" each need one.
- category: romantic (dating, partners, married, engaged, exes), family (parents, siblings, cousins, in-laws), friend (friends, including roommates, bandmates, teammates, climbing partners, and league mates, unless the turn says they are not friends), professional (coworkers, bosses, managers, employees, clients, business partners), or other (only acquaintances with no other tie).
- Exes are romantic with status past. If they stayed friends, add a second entry: friend, current.
- Two people whose wedding the turn mentions are married: romantic, current.
- label: a 1 to 3 word term, for example "college roommate", "ex-girlfriend", "cousin", "coworker".
- status: current or past. met_on: when they met, as a date, when the turn or the known facts tell or imply it (a couple who met at Coachella 2015 met in 2015). met_event: an event name.
- closeness 1 to 5, recency 1 to 5 (5 means in touch now), sentiment -2 to 2, only when the turn gives a signal.
- facts: the 0-based indexes of the facts in your output that support it.

Return empty lists when the turn has no new information.

EXAMPLE INPUT
KNOWN PEOPLE:
- Maria Lopez (aliases: Maria) [in the room]
- Tom Becker (aliases: Tom) [in the room]
- Felix Ward (aliases: Felix)
- Felix Shaw (aliases: Felix)
FACTS:
f4 told by Tom Becker: Tom Becker and Maria Lopez went to Lollapalooza in 2015.
NEW TURN (speaker: Maria Lopez):
No, we went to Lollapalooza in 2014, that is where we met. Felix drove us. My sister Ana was there too, she runs the Saturday book club with me.

EXAMPLE OUTPUT
{"speaker_name": null, "people": [{"name": "Ana", "gender": "female"}], "events": [{"name": "Lollapalooza 2014", "start_date": "2014", "people": ["speaker", "Tom Becker", "Ana"]}], "facts": [{"kind": "relationship", "text": "Maria Lopez and Tom Becker met at Lollapalooza in 2014.", "people": ["speaker", "Tom Becker"], "start_date": "2014", "event": "Lollapalooza 2014"}, {"kind": "story", "text": "Maria Lopez and Tom Becker went to Lollapalooza in 2014.", "people": ["speaker", "Tom Becker"], "start_date": "2014", "event": "Lollapalooza 2014", "disputes": "f4"}, {"kind": "story", "text": "Felix drove Maria Lopez and Tom Becker to Lollapalooza.", "people": ["Felix", "speaker", "Tom Becker"], "start_date": "2014", "event": "Lollapalooza 2014"}, {"kind": "membership", "text": "Ana and Maria Lopez run the Saturday book club.", "people": ["Ana", "speaker"], "group": "Saturday book club"}], "relationships": [{"a": "speaker", "b": "Tom Becker", "category": "friend", "label": "friend", "status": "current", "met_on": "2014", "met_event": "Lollapalooza 2014", "facts": [0]}, {"a": "speaker", "b": "Ana", "category": "family", "label": "sister", "status": "current", "closeness": 4, "recency": 5, "sentiment": 1, "facts": [3]}]}"""

NULLABLE = {"type": ["string", "null"]}
DATE = {"type": ["string", "null"], "description": "YYYY, YYYY-MM, or YYYY-MM-DD"}
REFS = {"type": "array", "items": {"type": "string"}}
SCORE = {"type": ["integer", "null"]}


def _object(optional: dict | None = None, **required: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": {**required, **(optional or {})},
    }


SCHEMA = _object(
    speaker_name=NULLABLE,
    people={
        "type": "array",
        "items": _object(
            {"aliases": REFS, "gender": {"type": "string", "enum": ["male", "female", "unknown"]}},
            name={"type": "string"},
        ),
    },
    events={
        "type": "array",
        "items": _object(
            {"id": NULLABLE, "when_text": NULLABLE, "start_date": DATE, "end_date": DATE, "place": NULLABLE},
            name={"type": "string"},
            people=REFS,
        ),
    },
    facts={
        "type": "array",
        "items": _object(
            {
                "when_text": NULLABLE,
                "start_date": DATE,
                "end_date": DATE,
                "event": NULLABLE,
                "group": NULLABLE,
                "revises": NULLABLE,
                "disputes": NULLABLE,
            },
            kind={"type": "string", "enum": ["personal", "relationship", "story", "membership", "correction"]},
            text={"type": "string"},
            people=REFS,
        ),
    },
    relationships={
        "type": "array",
        "items": _object(
            {
                "met_on": DATE,
                "met_event": NULLABLE,
                "closeness": SCORE,
                "recency": SCORE,
                "sentiment": SCORE,
                "facts": {"type": "array", "items": {"type": "integer"}},
            },
            a={"type": "string"},
            b={"type": "string"},
            category={"type": "string", "enum": ["romantic", "family", "friend", "professional", "other"]},
            label={"type": "string"},
            status={"type": "string", "enum": ["current", "past"]},
        ),
    },
)

DATE_PATTERN = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")
SEASONS = ("spring", "summer", "fall", "autumn", "winter")
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
MONTH_WORDS = re.compile(rf"\b({MONTHS}|month)\b|\d{{4}}-\d{{2}}")
DAY_WORDS = re.compile(
    rf"\b(today|tonight|yesterday|tomorrow|last night|this morning|days?|weeks?|weekend"
    rf"|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|\b({MONTHS}) \d{{1,2}}\b|\d{{4}}-\d{{2}}-\d{{2}}"
)
PERSON_NAME = re.compile(r"[A-Z][A-Za-z'.-]*(?: [A-Z][A-Za-z'.-]*){0,3}")
SPEAKER_REFS = {"speaker", "the speaker", "unknown speaker", "i", "me", "myself"}
RECENT_TURNS = 4
FUNCTION_WORDS = (
    {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of", "for", "with", "from", "by"}
    | {"is", "was", "are", "were", "he", "she", "his", "her", "they", "them", "their", "it", "s", "that"}
    | {"this", "after", "before", "when", "while", "used", "also", "too", "not", "no"}
)
COUPLE_EVENTS = {"wedding": "married", "engagement": "engaged"}
FRIEND_LABELS = ("bandmate", "roommate", "housemate", "teammate", "climbing partner", "league mate")
WORK_LABELS = ("coworker", "co-worker", "colleague", "manager", "boss", "at work")


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


@dataclass(frozen=True)
class Removed:
    """A name of a person whom the room deleted from the run. Facts and ties that name them are skipped."""


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
    """How precise a date is: never more precise than the words people said, such as "winter of 2022"."""
    if date is None:
        return None
    precision = {4: "year", 7: "month", 10: "day"}[len(date)]
    if not when_text:
        return precision
    said = when_text.lower()
    if DAY_WORDS.search(said):
        return precision
    if any(s in said for s in SEASONS):
        return "season"
    if MONTH_WORDS.search(said):
        return "year" if precision == "year" else "month"
    return "year"


def said_date(date: str | None, when_text: str | None) -> str | None:
    """A clean date cut to the precision that people said. The model turns "fall of 2016" into 2016-09."""
    date = clean_date(date)
    precision = date_precision(date, when_text)
    return date[: {"year": 4, "season": 4, "month": 7, "day": 10}[precision]] if date and precision else date


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
    room = set(graph.room())
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
        # A turn read again must not see its own facts, or the model finds nothing new and they all get hidden.
        if segment["id"] not in f["segment_ids"] and touches([p["person_id"] for p in f["people"]])
    ] or ["(none)"]
    lines += ["", "RECENT TURNS:"]
    lines += [f"{name(s['person_id'])}: {s['text']}" for s in recent] or ["(none)"]
    speaker = name(speaker_id) if speaker_id else f"unknown, voice {segment['speaker_label']}"
    if speaker_id is None and graph.phase == "intro":
        speaker += ", introducing themselves, so set speaker_name to the name they give"
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

    def person(ref: str) -> int | None | Ambiguity | Removed:
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
                case None | Removed():
                    continue
                case person_id:
                    ids.append(person_id)
        return list(dict.fromkeys(ids))

    waiting: dict[str, Ambiguity] = {}
    with store.batch():
        events = apply_events(store, output["events"], output["facts"], people, teller)

        for index, fact in enumerate(output["facts"]):
            if index in held.fact_ids:
                continue
            if any(isinstance(person(ref), Removed) for ref in fact["people"]):
                continue
            ids = people(fact["people"])
            if isinstance(ids, Ambiguity):
                waiting[ids.mention] = ids
                continue
            held.fact_ids[index] = apply_fact(store, fact, ids, events, teller)

        for index, rel in enumerate(output["relationships"]):
            if index in held.done_relationships:
                continue
            if any(isinstance(person(ref), Removed) for ref in (rel["a"], rel["b"])):
                continue
            ids = people([rel["a"], rel["b"]])
            if isinstance(ids, Ambiguity):
                waiting[ids.mention] = ids
                continue
            held.done_relationships.append(index)
            if len(ids) != 2:
                continue
            evidence = [held.fact_ids[i] for i in rel.get("facts") or [] if i in held.fact_ids]
            store.upsert_relationship(
                a=ids[0],
                b=ids[1],
                category=category_of(rel["category"], rel["label"]),
                fields={
                    "label": rel["label"],
                    "summary": store.graph.facts[evidence[0]]["text"] if evidence else None,
                    "status": rel["status"],
                    "met_on": clean_date(rel.get("met_on")) or met_on(store, ids, evidence),
                    "met_event_id": events.get(rel.get("met_event") or ""),
                    "closeness": rel.get("closeness"),
                    "recency": rel.get("recency"),
                    "sentiment": rel.get("sentiment"),
                },
                evidence=evidence,
                model_call_id=teller.model_call_id,
            )
        drop_revised_events(store, model_call_id=teller.model_call_id)

        for ambiguity in bare_first_names(store, text):
            if ambiguity.mention not in held.answers:
                waiting.setdefault(ambiguity.mention, ambiguity)

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


def category_of(category: str, label: str) -> str:
    """The prompts call bandmates, roommates, and teammates friends, yet the model files a band that plays
    paid gigs as professional. It also files a coworker as a friend at times, which splits the review votes."""
    if category in ("friend", "other") and any(word in label.lower() for word in WORK_LABELS):
        return "professional"
    if category == "professional" and any(word in label.lower() for word in FRIEND_LABELS):
        return "friend"
    return category


def met_on(store: Store, pair: list[int], evidence: list[int]) -> str | None:
    """The date of a fact that tells how the pair met: a supporting fact first, else any fact about both."""
    about_both = [f["id"] for f in store.graph.live_facts() if set(pair) <= {p["person_id"] for p in f["people"]}]
    for fact_id in [*evidence, *about_both]:
        fact = store.graph.facts[fact_id]
        if fact["start_date"] and re.search(r"\bmet\b|\bmeet\b", fact["text"].lower()):
            return fact["start_date"]
    return None


def resolve(
    store: Store, ref: str, entry: Record | None, *, teller: Teller, answers: dict[str, int], text: str
) -> int | None | Ambiguity | Removed:
    """Finds the person a model ref names, creates a new person, or reports that the name fits several people."""
    if ref.strip().lower() in SPEAKER_REFS:
        return teller.person_id
    ref = drop_ly(ref)
    entry = {**entry, "name": drop_ly(entry["name"])} if entry else None
    if re.search(rf"\b(?:dog|cat|puppy|kitten|pet)s? {re.escape(ref)}\b", text):
        return None
    if ref in answers:
        return answers[ref]
    matches = in_run_first(store, name_matches(store.conn, ref))
    if not matches and entry:
        matches = in_run_first(store, name_matches(store.conn, entry["name"]))
    name = (entry["name"] if entry else ref).strip()
    gender = entry.get("gender") or "unknown" if entry else "unknown"
    if matches and matches[0]["id"] not in store.graph.people and near_in_run(store, name, gender):
        matches = []
    if any(p["id"] in store.graph.removed for p in matches):
        return Removed()
    if len(matches) > 1:
        return Ambiguity(ref, [p["id"] for p in matches])
    if not matches:
        if not PERSON_NAME.fullmatch(name) or name.split()[0] == "The":
            return None
        aliases = entry.get("aliases") or [] if entry else []
        if " " in name:
            near = near_in_run(store, name, gender)
            if len(near) == 1:
                return respelled(store, near[0], name)
            if len(near) > 1:
                return Ambiguity(ref, [p["id"] for p in near])
            first = name.split()[0]
            known = first_name_person(store, first)
            if known in store.graph.removed:
                return Removed()
            if known is not None:
                store.update_person(known, add_aliases=[first, *aliases], gender=gender, name=name)
                return known
            aliases = [first, *aliases] if first not in aliases else aliases
        else:
            near = near_in_run(store, name, gender)
            if any(p["id"] in store.graph.removed for p in near):
                return Removed()
            if len(near) > 1:
                return Ambiguity(ref, [p["id"] for p in near])
            if near:
                store.update_person(near[0]["id"], add_aliases=[name], gender=None)
                return near[0]["id"]
        return store.create_person(name=name, aliases=aliases, gender=gender)["id"]

    person = matches[0]
    if ambiguity := first_name_only(store, person, text):
        return answers.get(ambiguity.mention, ambiguity)
    if entry:
        store.update_person(person["id"], add_aliases=entry.get("aliases") or [], gender=entry.get("gender"))
    return person["id"]


def near_in_run(store: Store, name: str, gender: str) -> list[Record]:
    """The people of this run whom `name` could be a misspelling of. A leftover misspelling from another
    session, such as "Gretta", must not win over the Greta this run talks about."""
    if " " in name:
        return [p for p in store.graph.people.values() if misspelled(p["name"], name)]
    return near_people(store, name, gender)


def near_people(store: Store, heard: str, gender: str) -> list[Record]:
    """The people of this run whose first name or a one-word alias the lone name `heard` is a near miss of. A
    person of another gender is never one."""
    room = set(store.graph.room())
    return [
        p
        for p in store.graph.people.values()
        if p["merged_into_id"] is None
        and (gender == p["gender"] or {gender, p["gender"]} & {"unknown", None})
        and any(
            near_first(heard, n, room=p["id"] in room) for n in [p["name"].split()[0], *p["aliases"]] if " " not in n
        )
    ]


def respell_people(store: Store, text: str) -> None:
    """Checks the full names in a new turn against the people of the run. The model writes a known person by
    the known spelling, so "Kate Russo, not Kate Miller" would never correct an earlier "Kay Russo"."""
    for heard in pairs_of(text):
        near = [p for p in store.graph.people.values() if p["merged_into_id"] is None and misspelled(p["name"], heard)]
        if len(near) == 1 and near[0]["name"] != heard:
            respelled(store, near[0], heard)


def respelled(store: Store, person: Record, heard: str) -> int:
    """Keeps both spellings of a person whom speech recognition spelled two ways, and names them by the better
    one. A name typed into the roster stays."""
    said = " ".join(s["text"] for s in store.graph.segments.values())
    if better_spelling(heard, person["name"], said) == heard and person["id"] not in store.graph.roster:
        store.update_person(person["id"], add_aliases=[person["name"], heard.split()[0]], gender=None, name=heard)
    else:
        store.update_person(person["id"], add_aliases=[heard], gender=None)
    return person["id"]


def in_run_first(store: Store, matches: list[Record]) -> list[Record]:
    """Narrows name matches to the people this run already talked about, when any match is one of them.
    People are global, so "Andrew" would otherwise fit every Andrew from every session. A name that fits no
    one in this run and several people of other sessions, such as a first name, is a new person of this run,
    because the room cannot say which person of another session it means."""
    in_run = [p for p in matches if p["id"] in store.graph.people]
    return in_run or (matches if len({p["name"] for p in matches}) == 1 else [])


def first_name_person(store: Store, first: str) -> int | None:
    """The one person known only by this first name, who a full name such as "Tyler Grant" now completes."""
    rows = store.conn.execute(
        "SELECT id FROM people WHERE merged_into_id IS NULL AND lower(name) = lower(?)", (first,)
    ).fetchall()
    return rows[0][0] if len(rows) == 1 and len(name_matches(store.conn, first)) == 1 else None


def bare_first_names(store: Store, text: str) -> list[Ambiguity]:
    """First names in a turn that fit two or more people of this run, with no full name or other alias that
    says which one. The model skips questions such as "Does Megan know Kate is coming?", yet the room still
    has to say which Kate."""
    said = set(re.findall(r"[a-z']+", text.lower()))
    by_first: dict[str, list[Record]] = {}
    for person in store.graph.people.values():
        if person["merged_into_id"] is None:
            by_first.setdefault(person["name"].split()[0].lower(), []).append(person)
    found = []
    for first, people in by_first.items():
        if len(people) < 2 or first not in said:
            continue
        names = [n for p in people for n in [p["name"], *p["aliases"]] if n.lower() != first]
        if not any(set(n.lower().split()) <= said for n in names):
            found.append(Ambiguity(people[0]["name"].split()[0], [p["id"] for p in people]))
    return found


def first_name_only(store: Store, person: Record, text: str) -> Ambiguity | None:
    """The model may pick one Andrew when the turn only says "Andrew". The room decides instead."""
    first = person["name"].split()[0]
    others = in_run_first(store, name_matches(store.conn, first))
    if len(others) < 2:
        return None
    said = set(re.findall(r"[a-z']+", text.lower()))
    shared = {first.lower()}
    distinct = [n for n in [person["name"], *person["aliases"]] if n.lower() not in shared]
    if first.lower() not in said or any(set(n.lower().split()) <= said for n in distinct):
        return None
    return Ambiguity(first, [p["id"] for p in others])


def apply_events(store: Store, events: list[Record], facts: list[Record], people, teller: Teller) -> dict[str, int]:
    """Writes the events of an output and maps each event ref to its id. The model sometimes lists a named
    group as an event, so an event named after a group is skipped. A fact can name an event that the output
    does not list, which then takes the fact's date."""
    ids: dict[str, int] = {}
    groups = [g["name"].lower() for g in store.graph.groups.values()] + [
        f["group"].lower() for f in facts if f.get("group")
    ]
    for event in events:
        if any(group in event["name"].lower() for group in groups):
            continue
        event_id = store.graph.lookup("e", event.get("id") or event["name"])
        if event_id is None:
            event_id = next(
                (e["id"] for e in store.graph.events.values() if e["name"].lower() == event["name"].lower()), None
            )
        attendees = people(event["people"])
        start = said_date(event.get("start_date"), event.get("when_text")) or year_in(event["name"])
        record = store.upsert_event(
            event_id,
            {
                "name": event["name"] if store.graph.lookup("e", event["name"]) is None else None,
                "when_text": event.get("when_text"),
                "start_date": start,
                "end_date": clean_date(event.get("end_date")),
                "date_precision": date_precision(start, event.get("when_text")),
                "place": event.get("place"),
            },
            people=attendees if isinstance(attendees, list) else [],
            model_call_id=teller.model_call_id,
        )
        ids[event["name"]] = record["id"]
        couple_of(store, record, model_call_id=teller.model_call_id)
        if event.get("id"):
            ids[event["id"]] = record["id"]
    for event_id in store.graph.events:
        ids.setdefault(store.graph.ref("e", event_id), event_id)
    for fact in facts:
        name = fact.get("event")
        if not name or name in ids or any(group in name.lower() for group in groups):
            continue
        known = store.graph.lookup("e", name) or next(
            (e["id"] for e in store.graph.events.values() if e["name"].lower() == name.lower()), None
        )
        start = said_date(fact.get("start_date"), fact.get("when_text")) or year_in(name)
        fields = {
            "name": name,
            "when_text": fact.get("when_text"),
            "start_date": start,
            "date_precision": date_precision(start, fact.get("when_text")),
        }
        ids[name] = known or store.upsert_event(None, fields, people=[], model_call_id=teller.model_call_id)["id"]
    return ids


def couple_of(store: Store, event: Record, *, model_call_id: int | None) -> None:
    """Makes the two people a wedding or an engagement is named after a couple, such as "Arjun Mehta and
    Claire Dunn's wedding". The model names the event after the couple, yet often leaves out the tie when
    the speaker is a guest."""
    label = next((label for word, label in COUPLE_EVENTS.items() if word in event["name"].lower()), None)
    if label is None:
        return
    words = set(re.findall(r"[a-z]+", event["name"].lower().replace("'s", " ")))
    couple = [
        p["id"]
        for p in store.graph.people.values()
        if p["merged_into_id"] is None and p["name"].lower().split()[0] in words
    ]
    if len(couple) != 2 or any(
        {r["a"], r["b"]} == set(couple) and r["category"] == "romantic" for r in store.graph.relationships.values()
    ):
        return
    fields = {"label": label, "status": "current"}
    store.upsert_relationship(
        a=couple[0], b=couple[1], category="romantic", fields=fields, evidence=[], model_call_id=model_call_id
    )


def apply_fact(store: Store, fact: Record, people: list[int], events: dict[str, int], teller: Teller) -> int:
    start = said_date(fact.get("start_date"), fact.get("when_text"))
    event_id = events.get(fact.get("event") or "")
    roles = [(pid, "subject") for pid in people]
    if teller.person_id is not None and teller.person_id not in people:
        roles.append((teller.person_id, "teller"))
    record = store.add_fact(
        {
            "kind": fact["kind"],
            "text": fact["text"],
            "told_by_person_id": teller.person_id,
            "when_text": fact.get("when_text"),
            "start_date": start,
            "end_date": clean_date(fact.get("end_date")),
            "date_precision": date_precision(start, fact.get("when_text")),
            "event_id": event_id,
            "confidence": teller.confidence,
        },
        people=roles,
        segment_ids=[teller.segment_id],
        model_call_id=teller.model_call_id,
    )
    if event_id is not None and people:
        store.upsert_event(event_id, {}, people=people, model_call_id=teller.model_call_id)

    if fact["kind"] == "membership" and fact.get("group") and proper_group_name(fact["group"]):
        group = store.group_for(fact["group"])
        for person_id in people:
            store.add_member(
                group_id=group["id"],
                person_id=person_id,
                since=start,
                until=clean_date(fact.get("end_date")),
                fact_id=record["id"],
            )

    old = store.graph.lookup("f", fact.get("revises")) or store.graph.lookup("f", fact.get("disputes"))
    if old is None:
        return record["id"]
    told_by = store.graph.facts[old]["told_by_person_id"]
    related = same_thing(store.graph.facts[old], store.graph.facts[record["id"]])
    # The model also sets "revises" on news, such as a planned move against where someone lives.
    own = told_by == teller.person_id if told_by is not None else not fact.get("disputes")
    if own and (fact["kind"] == "correction" or related):
        revise(store, old, record["id"], model_call_id=teller.model_call_id)
    # "It was raining!" after "Felix crashed the car" is an excuse. Only a correction or another date disputes.
    elif not own and related and (fact["kind"] == "correction" or dates_differ(store.graph.facts[old], record)):
        dispute(store, old, record["id"], model_call_id=teller.model_call_id)
    return record["id"]


def same_thing(a: Record, b: Record) -> bool:
    """Whether two facts can disagree: they share a common word, not a name or a place, and either their dates
    differ or two and at least half the common words of the shorter one are in both and some word differs. The model marks a
    new story or a retelling as a dispute at times, such as "drove Felix to the hospital" against "ate Felix's
    lunch", or "shared a dorm in Lisbon" against "stole a parking spot in Lisbon"."""
    ignored = FUNCTION_WORDS | {"speaker"}

    def common(text: str) -> set[str]:
        return set(re.findall(r"\b[a-z0-9]+\b", text)) - ignored

    first, second = common(a["text"]), common(b["text"])
    shared = len(first & second)
    if shared == 0:
        return False
    if dates_differ(a, b):
        return True
    return shared >= 2 and 2 * shared >= min(len(first), len(second)) and a["text"].lower() != b["text"].lower()


def dates_differ(a: Record, b: Record) -> bool:
    dates = sorted([f["start_date"] or year_in(f["text"]) or "" for f in (a, b)], key=len)
    return bool(dates[0]) and not dates[1].startswith(dates[0])


def revise(store: Store, old: int, new: int, *, model_call_id: int) -> None:
    """A teller corrects their own fact, or agrees with a correction. That settles a dispute about the fact."""
    store.update_fact(old, revised_by_fact_id=new, model_call_id=model_call_id)
    for prompt in store.graph.active_prompts():
        if prompt["reason"] == "conflict" and old in prompt["payload"]["fact_ids"]:
            store.dismiss_prompt(prompt["id"])
            for fact_id in prompt["payload"]["fact_ids"]:
                if fact_id != old and store.graph.facts[fact_id]["disputed"]:
                    store.update_fact(fact_id, disputed=0, model_call_id=model_call_id)


async def read_again(
    store: Store, llm: LLM, segment_id: int, *, today: str, model: str, seed: int, context: Context
) -> bool:
    """Extracts a turn again with its current speaker and text, and returns whether the model answered. The new
    facts replace the old facts of the turn."""
    segment = store.graph.segments[segment_id]
    old = [f["id"] for f in store.graph.live_facts() if segment_id in f["segment_ids"]]
    call = extraction_call(
        store, segment=segment, speaker_id=segment["person_id"], today=today, model=model, seed=seed, context=context
    )
    watch = watch_on(store, worker="extractor", segment_id=segment_id)
    result = await llm.call(call, run_id=store.run_id, audio_ms=segment["end_ms"], watch=watch)
    if result.output is None:
        return False
    teller = Teller(segment_id, segment["person_id"], segment["speaker_confidence"] or 1.0, result.model_call_id)
    held = Held(result.output, teller)
    apply_output(store, held)
    retell(store, old, list(held.fact_ids.values()), model_call_id=result.model_call_id)
    return True


def retell(store: Store, old: list[int], new: list[int], *, model_call_id: int) -> None:
    """Replaces the facts of a turn read again under a new teller. The old facts close their conflicts, and a
    relationship that only the old facts showed goes away, because "my cousin" named the wrong person. With no
    new fact, an old fact revises itself, which hides it like any revised fact. A new fact that still
    disagrees with the other side of a conflict keeps the dispute."""
    for fact_id in old:
        rivals = [
            other
            for prompt in store.graph.active_prompts()
            if prompt["reason"] == "conflict" and fact_id in prompt["payload"]["fact_ids"]
            for other in prompt["payload"]["fact_ids"]
            if other != fact_id
        ]
        revise(store, fact_id, new[0] if new else fact_id, model_call_id=model_call_id)
        for rival in rivals:
            told_by = store.graph.facts[rival]["told_by_person_id"]
            match = next(
                (
                    n
                    for n in new
                    if store.graph.facts[n]["told_by_person_id"] != told_by
                    and same_thing(store.graph.facts[rival], store.graph.facts[n])
                ),
                None,
            )
            if match is not None:
                dispute(store, rival, match, model_call_id=model_call_id)
    for relationship in list(store.graph.relationships.values()):
        if relationship["evidence"] and set(relationship["evidence"]) <= set(old):
            store.remove_relationship(relationship["id"])
    drop_revised_events(store, model_call_id=model_call_id)


def drop_revised_events(store: Store, *, model_call_id: int | None) -> None:
    """Removes an event that only revised facts tell of, such as "Electric Zoo 2019" after the teller gives in
    to 2018. It runs after a whole output, because a later fact of the output can still name the event."""
    live = {f["event_id"] for f in store.graph.live_facts()}
    revised = {f["event_id"] for f in store.graph.facts.values() if f["revised_by_fact_id"] is not None}
    for event_id in revised - live - {None}:
        store.remove_event(event_id, model_call_id=model_call_id)


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
