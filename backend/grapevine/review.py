"""The review pass: the model reads the whole transcript beside the record and completes the relationships,
group members, and events that single turns left out. One turn rarely says that two housemates are friends
or who is on a team. The whole conversation does.

Three calls run at once, one for each kind of record, because one call asked for all of them drops items.
Each call writes short notes before its list, which the list then follows."""

import asyncio
import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from grapevine.extract import (
    Ambiguity,
    Teller,
    category_of,
    clean_date,
    couple_of,
    date_precision,
    resolve,
    year_in,
)
from grapevine.graph import Record
from grapevine.llm import LLM, Call, watch_on
from grapevine.names import misspelled
from grapevine.store import GROUP_WORDS, Store, group_key, proper_group_name

PROMPT_VERSION = "review-v20"
SAME_EVENT = 0.6
# The prompt says that how two people met, or that they dated, is not a happening, and the model still lists it.
NOT_HAPPENINGS = {"relationship", "meeting", "met", "dating", "friendship"}
EVENT_KINDS = {"the", "and", "with", "trip", "party", "event", "wedding", "birthday", "move", "visit", "weekend"}
VOTES = 3
LINE_MATCH = 0.8
VOTE_TEMPERATURE = 0.5

INTRO = """You check the record of a group conversation against its full transcript, and complete it. Return JSON only.

The input lists today's date, the people (the ones in the room are marked), the groups, the events, and the relationships in the record, then the numbered transcript with the speaker of each turn. Refer to people by the full names in the record. Write a new person's name as said.

"""

RELATIONSHIPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: one line for each other pair the conversation ties together, such as a couple whose wedding someone mentions, a sibling, a cousin, an ex, or a coworker. Include pairs where neither person is in the room: "my cousin's wedding, Tom and Rachel" makes Tom and Rachel a married couple. Then fill "room", which has one key for each pair of people in the room, and "other", for every pair in your analysis.

- For each pair in "room", list its relationships, or an empty list when the conversation does not show how these two know each other. Being at the same party or in the same room is not a tie.
- List every relationship the conversation states or clearly implies, including the ones in the record. Fix a record entry when the transcript shows a better category, status, label, or meeting date.
- Infer what the conversation implies. People who share a house, or hold a house meeting, are roommates and friends. A couple who is married or planning a wedding is romantic, and two people whose wedding someone went to are married. Exes who are still in each other's lives, for example one still helps the other move house, are also friends.
- category:
  - romantic: dating, partners, engaged, married, exes.
  - family: parents, children, siblings, cousins, in-laws.
  - friend: friends, including roommates, bandmates (also in a band that plays paid gigs), teammates, climbing partners, league mates, and co-founders of a league or club, unless the conversation says they are not friends.
  - professional: coworkers, bosses, managers, employees, clients, business partners.
  - other: only acquaintances with no other tie.
- A pair can have two entries. Exes who stayed friends get romantic with status past and friend with status current.
- status: current or past, for the tie itself. Exes are romantic and past.
- in_touch: true when the two are still in each other's lives today, for example old roommates who still meet, a friend who gave a toast at the other's recent wedding, or an ex who still helps the other move house. False when the conversation says they lost touch or one died.
- met_on: when the pair met, as "YYYY" or "YYYY-MM", when the conversation tells or implies it. A couple who met at Coachella in April 2016 has "2016-04". If the conversation only says when the relationship began, such as "we dated from 2020 to 2022", use that. Coworkers met when the second one started the job. Else null.
- label: a 1 to 3 word term, for example "college roommate", "ex-girlfriend", "cousin". summary: one sentence."""
)

GROUPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: for each named group (a crew, a team, a band, a club, a league) that anyone names, even once, such as a league someone runs, the plain phrases people use for it, such as "the crew" or "the league", and each line of the transcript that puts someone in it under any of those names. Then fill "known", which has one key for each group in the record, and "new", for named groups the record does not have.

- For each known group, list its plain phrases in "also_called", and answer for each person whether they are in it under any of its names. People who say "we", "our team", or "our league" about the group are in it, and so are the people they list as going with "the whole crew".
- Every named group in your analysis that is not under "known" goes in "new", with one entry under its proper name, for example "Thursday Flyers", and everyone in it.
- A house, a meeting, or an event is not a group."""
)

MEMBERS_SYSTEM = """You read the lines of a group conversation that mention one group, and list who is in it. Return JSON only.

Write "analysis" first: for each line, who it puts in the group. Then list "members" with the full names under PEOPLE when they match.
- The speaker of a line is in the group when they say "we", "our", or "me" about it, or call themselves a member.
- The speaker of a line is in the group when they talk about it from the inside, such as how it is run or how it did, even without naming it or saying "we". "Which she runs like a tyrant", said about the club's president, puts the speaker in the club. "The team lost again, and I am livid" puts the speaker in the team.
- Asking whether someone goes with the group, or telling who leads it, does not by itself put the speaker in it.
- People a line names as going with the whole group are in it, for example "the whole crew is going: me, Dev, and Lena".
- A line can answer the line before it, for example "Are you going with the Thursday Flyers crew?" "Yes, me, Dev, and Lena."
- People a line calls members are in it. Speech recognition misspells group names, so "Thursday Fliers" is "Thursday Flyers".
- Speech recognition also mishears first names. A name in a line that sounds like exactly one person under PEOPLE, such as "Erin" for "Aaron Cole", is that person."""

EVENTS_SYSTEM = (
    INTRO
    + """Write "analysis" first: go through the transcript in order, and write one line for every happening that the conversation gives a time for (a year, a season, "last weekend"), whether or not the record has it. Look for trips, weddings (also one said in passing, such as "Rosa got married in 2021, so it couldn't be then"), engagements, parties, accidents, injuries, shows, moves to a new city, and moves into a house. A time can come in the next line, for example "Greta left me for Seattle." "In 2017." Keep each line under 25 words, and do not argue with yourself. Then list each one in "happenings".

- List every happening of your analysis, including one whose date people dispute.
- Not happenings: how or when two people met, a relationship, a job, school years, a visit planned for the coming days, or a game result.
- A show, a party, or a game that people go to today or tonight is a happening, dated today.
- record: the record event that is the same happening, or null. A move is never a birthday or a visit.
- start_date: "YYYY", "YYYY-MM", or "YYYY-MM-DD", resolved with today's date when the conversation says "last weekend" or "tonight". When people disagree on a date, use the date of the person who was there, or the date that the reasons given support, for example "Rosa got married in 2021, so the trip was 2022".
- name: what and who, for example "Greta's move to Seattle" or "Rosa's wedding". people: who took part."""
)

NULLABLE = {"type": ["string", "null"]}
NAMES = {"type": "array", "items": {"type": "string"}}
TEXT = {"type": "string"}


def _object(**properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


TIE = {
    "category": {"type": "string", "enum": ["romantic", "family", "friend", "professional", "other"]},
    "label": TEXT,
    "status": {"type": "string", "enum": ["current", "past"]},
    "met_on": NULLABLE,
    "in_touch": {"type": "boolean"},
    "summary": TEXT,
}


def relationships_schema(pairs: list[str]) -> dict:
    """One required key for each pair in the room, so the model answers for every pair."""
    return _object(
        analysis=TEXT,
        room=_object(**{pair: {"type": "array", "items": _object(**TIE)} for pair in pairs}),
        other={"type": "array", "items": _object(a=TEXT, b=TEXT, **TIE)},
    )


def groups_schema(groups: list[str], people: list[str]) -> dict:
    """One required answer for each person in each known group, so the model decides about everyone."""
    member = _object(also_called=NAMES, members=_object(**{name: {"type": "boolean"} for name in people}))
    return _object(
        analysis=TEXT,
        known=_object(**{group: member for group in groups}),
        new={"type": "array", "items": _object(name=TEXT, also_called=NAMES, members=NAMES)},
    )


def events_schema(record: list[str]) -> dict:
    """Every dated happening, each tied to the record event it is, so one list both adds events and corrects
    record dates."""
    happening = _object(
        name=TEXT,
        record={"type": ["string", "null"], "enum": [*record, None]},
        start_date=NULLABLE,
        place=NULLABLE,
        people=NAMES,
    )
    return _object(analysis=TEXT, happenings={"type": "array", "items": happening})


MEMBERS_SCHEMA = _object(analysis=TEXT, members=NAMES)


def user_message(store: Store, *, today: str) -> str:
    graph = store.graph
    room = graph.room()
    names = {pid: p["name"] for pid, p in graph.people.items()}
    lines = [f"TODAY: {today}", "", "PEOPLE:"]
    lines += [
        f"- {p['name']}{' [in the room]' if pid in room else ''}"
        for pid, p in graph.people.items()
        if p["merged_into_id"] is None
    ]
    lines += ["", "GROUPS:"]
    members = defaultdict(set)
    for member in graph.members.values():
        members[member["group_id"]].add(names.get(member["person_id"], "?"))
    lines += [f"- {g['name']}: {', '.join(sorted(members[gid])) or 'no members'}" for gid, g in graph.groups.items()]
    lines += ["", "EVENTS:"]
    lines += [f"- {e['name']} ({e['start_date'] or 'no date'})" for e in graph.events.values()]
    lines += ["", "RELATIONSHIPS:"]
    lines += [
        f'- {names[r["a"]]} - {names[r["b"]]}: {r["category"]} "{r["label"]}" {r["status"]}'
        + (f", met {r['met_on']}" if r["met_on"] else "")
        for r in graph.relationships.values()
    ]
    lines += ["", "TRANSCRIPT:"]
    turns = sorted(graph.segments.values(), key=lambda s: s["start_ms"])
    lines += [f"{i} {names.get(s['person_id'], 'Unknown speaker')}: {s['text']}" for i, s in enumerate(turns, 1)]
    return "\n".join(lines)


async def review(store: Store, llm: LLM, *, model: str, seed: int, today: str) -> None:
    if not store.graph.segments:
        return
    user = user_message(store, today=today)
    names = store.graph.people
    pairs = {f"{names[a]['name']} and {names[b]['name']}": (a, b) for a, b in combinations(store.graph.room(), 2)}
    people = list(dict.fromkeys(p["name"] for p in names.values() if p["merged_into_id"] is None))
    record_events = {e["name"]: e["id"] for e in reversed(list(store.graph.events.values()))}

    def ask(system: str, schema: dict, *, seed: int = seed, temperature: float = 0) -> Call:
        return Call("reviewer", model, PROMPT_VERSION, system, user, schema, seed, 6000, temperature)

    watch = watch_on(store, worker="reviewer")
    votes = [
        ask(RELATIONSHIPS_SYSTEM, relationships_schema(list(pairs)), seed=seed + i, temperature=VOTE_TEMPERATURE)
        for i in range(VOTES)
    ]
    groups_call = ask(GROUPS_SYSTEM, groups_schema([g["name"] for g in store.graph.groups.values()], people))
    groups, events, *voted = await asyncio.gather(
        llm.call(groups_call, run_id=store.run_id, audio_ms=store.audio_ms),
        llm.call(ask(EVENTS_SYSTEM, events_schema(list(record_events))), run_id=store.run_id, audio_ms=store.audio_ms),
        *[
            llm.call(vote, run_id=store.run_id, audio_ms=store.audio_ms, watch=watch if i == 0 else None)
            for i, vote in enumerate(votes)
        ],
    )

    found_groups = []
    if groups.output:
        found_groups = [
            {
                "name": name,
                "also_called": answer["also_called"],
                "members": [p for p, y in answer["members"].items() if y],
            }
            for name, answer in groups.output["known"].items()
        ] + groups.output["new"]
        more = await asyncio.gather(*[members_of(store, llm, g, model=model, seed=seed) for g in found_groups])
        found_groups = [{**g, "members": [*g["members"], *extra]} for g, extra in zip(found_groups, more)]

    def person(name: str, model_call_id: int) -> int | None:
        known = [p for p in store.graph.people.values() if misspelled(p["name"], name)]
        if len(known) == 1 and name not in {p["name"] for p in store.graph.people.values()}:
            return known[0]["id"]
        teller = Teller(segment_id=-1, person_id=None, confidence=1.0, model_call_id=model_call_id)
        found = resolve(store, name, None, teller=teller, answers={}, text="")
        return None if isinstance(found, Ambiguity) else found

    with store.batch():
        outputs = [v.output for v in voted if v.output]
        if outputs:
            room = [
                [
                    {**tie, "a": names[pairs[pair][0]]["name"], "b": names[pairs[pair][1]]["name"]}
                    for pair, ties in output["room"].items()
                    for tie in ties
                ]
                for output in outputs
            ]
            call_id = voted[0].model_call_id
            needed = len(room) // 2 + 1
            apply_relationships(store, agreed(room, person, call_id, needed=needed), person, call_id, prune=True)
            other = agreed([o["other"] for o in outputs], person, call_id, needed=needed)
            apply_relationships(store, other, person, call_id, prune=False)
        if groups.output:
            apply_groups(store, found_groups, person, groups.model_call_id)
        if events.output:
            apply_happenings(store, events.output["happenings"], record_events, person, events.model_call_id)


async def members_of(store: Store, llm: LLM, group: Record, *, model: str, seed: int) -> list[str]:
    """Asks who is in one group, with only the lines about it. In the whole transcript, the model misses a
    line such as "the whole crew is going: me, Owen, and Kevin"."""
    names = {pid: p["name"] for pid, p in store.graph.people.items()}
    turns = sorted(store.graph.segments.values(), key=lambda s: s["start_ms"])
    picked = group_lines([s["text"] for s in turns], [group["name"], *group["also_called"]])
    if not picked:
        return []
    lines = [f"{names.get(turns[i]['person_id'], 'Unknown speaker')}: {turns[i]['text']}" for i in picked]
    user = "\n".join(
        [f"GROUP: {group['name']} (also called {', '.join(group['also_called']) or 'nothing else'})", "", "PEOPLE:"]
        + [f"- {name}" for name in names.values()]
        + ["", "LINES:", *lines]
    )
    call = Call("reviewer", model, PROMPT_VERSION, MEMBERS_SYSTEM, user, MEMBERS_SCHEMA, seed, max_tokens=1500)
    result = await llm.call(call, run_id=store.run_id, audio_ms=store.audio_ms)
    return result.output["members"] if result.output else []


def group_lines(texts: list[str], group_names: list[str]) -> list[int]:
    """The indexes of the lines about a group: lines that name it under any spelling close to one of its
    names, the two lines after each, which often answer it, and lines that use its plain noun, such as "the
    whole crew"."""

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    keys = [words(group_key(name)) for name in group_names if group_key(name)]

    def names_it(text: str) -> bool:
        said = words(text)
        return any(
            SequenceMatcher(None, " ".join(said[i : i + len(key)]), " ".join(key)).ratio() >= LINE_MATCH
            for key in keys
            for i in range(len(said) - len(key) + 1)
        )

    named = [i for i, text in enumerate(texts) if names_it(text)]
    near = {j for i in named for j in range(i, min(i + 3, len(texts)))}
    nouns = {w for i in near for w in words(texts[i]) if w in GROUP_WORDS} | {
        w for name in group_names for w in words(name) if w in GROUP_WORDS
    }
    return sorted(near | {i for i, text in enumerate(texts) if nouns & set(words(text))})


def agreed(lists: list[list[Record]], person, model_call_id: int, *, needed: int) -> list[Record]:
    """The relationships that at least `needed` votes list, by pair and category. Each vote samples the model
    at a small temperature, so one vote's slip, such as calling two guests friends or a coworker a friend,
    is outvoted. An acquaintance entry is dropped when the pair has a closer tie."""
    seen: dict[tuple[frozenset, str], list[Record]] = defaultdict(list)
    for entries in lists:
        for entry in {
            (frozenset((entry["a"], entry["b"])), entry["category"]): entry for entry in ties(entries)
        }.values():
            a, b = person(entry["a"], model_call_id), person(entry["b"], model_call_id)
            if a is not None and b is not None and a != b:
                seen[(frozenset((a, b)), entry["category"])].append(entry)
    kept = {key: entries for key, entries in seen.items() if len(entries) >= needed}
    closer = {pair for pair, category in kept if category != "other"}
    return [
        {**entries[0], "met_on": next((e["met_on"] for e in entries if e["met_on"]), None)}
        for (pair, category), entries in kept.items()
        if category != "other" or pair not in closer
    ]


def ties(entries: list[Record]) -> list[Record]:
    """The relationships that each listed tie stands for. A tie between people still in touch is current, and
    exes still in touch are also friends."""
    result = []
    for entry in entries:
        entry = {**entry, "category": category_of(entry["category"], entry["label"])}
        if entry["in_touch"] and entry["category"] == "romantic" and entry["status"] == "past":
            result += [entry, {**entry, "category": "friend", "label": "friend", "status": "current", "met_on": None}]
        elif entry["in_touch"]:
            result.append({**entry, "status": "current"})
        else:
            result.append(entry)
    return result


def apply_relationships(store: Store, entries: list[Record], person, model_call_id: int, *, prune: bool) -> None:
    """Writes each listed relationship. With `prune`, for a pair the review lists, it also removes the
    categories the review left out, because the review read the whole conversation and a single turn did not.
    The review answers every pair in the room, so only those answers prune. A pair outside the room is one
    the model chose to list, and it can miss a category that a turn got right."""
    kept: dict[frozenset, set[str]] = defaultdict(set)
    for entry in entries:
        a, b = person(entry["a"], model_call_id), person(entry["b"], model_call_id)
        if a is None or b is None or a == b:
            continue
        kept[frozenset((a, b))].add(entry["category"])
        store.upsert_relationship(
            a=a,
            b=b,
            category=entry["category"],
            fields={
                "label": entry["label"],
                "summary": entry["summary"],
                "status": entry["status"],
                "met_on": clean_date(entry["met_on"]),
            },
            evidence=[],
            model_call_id=model_call_id,
        )
    for relationship in list(store.graph.relationships.values()):
        categories = kept.get(frozenset((relationship["a"], relationship["b"])))
        if prune and categories and relationship["category"] not in categories:
            store.remove_relationship(relationship["id"])


def apply_groups(store: Store, entries: list[Record], person, model_call_id: int) -> None:
    for entry in [e for e in entries if proper_group_name(e["name"])]:
        group = store.group_for(entry["name"])
        known = {m["person_id"] for m in store.graph.members.values() if m["group_id"] == group["id"]}
        found = [person(name, model_call_id) for name in entry["members"]]
        new = list(dict.fromkeys(pid for pid in found if pid is not None and pid not in known))
        if not new:
            continue
        names = [store.person(pid)["name"] for pid in new]
        fact = store.add_fact(
            {
                "kind": "membership",
                "text": f"{', '.join(names)} {'is' if len(new) == 1 else 'are'} in {group['name']}.",
            },
            people=[(pid, "subject") for pid in new],
            segment_ids=[],
            model_call_id=model_call_id,
        )
        for pid in new:
            store.add_member(group_id=group["id"], person_id=pid, since=None, until=None, fact_id=fact["id"])


def apply_happenings(
    store: Store, happenings: list[Record], record: dict[str, int], person, model_call_id: int
) -> None:
    """Moves each record event to the date of the happening linked to it, and adds the other happenings.
    The model links a happening to a wrong record event at times, so a link needs a shared word, and only the
    first happening linked to a record event keeps the link. "Nina's move to Denver" must not move the date
    of "Move to Denver 2016", which is Megan's."""
    linked = set()
    added = []
    for happening in happenings:
        start = clean_date(happening["start_date"])
        event = store.graph.events.get(record.get(happening["record"] or ""))
        if event is None or event["id"] in linked or not same_kind(store, happening["name"], event["name"]):
            words = set(re.findall(r"[a-z]+", happening["name"].lower()))
            if words and not words & NOT_HAPPENINGS:
                added.append(happening)
            continue
        linked.add(event["id"])
        if start and start != event["start_date"]:
            fields = {"start_date": start, "date_precision": date_precision(start, None)}
            store.upsert_event(event["id"], fields, people=[], model_call_id=model_call_id)
    apply_events(store, added, person, model_call_id)


def apply_events(store: Store, entries: list[Record], person, model_call_id: int) -> None:
    """Adds each new event, and moves a record event to the date the review gives it."""
    for entry in entries:
        start = clean_date(entry["start_date"]) or year_in(entry["name"])
        known = next((e for e in store.graph.events.values() if same_event(entry["name"], start, e)), None)
        if known and (start is None or known["start_date"] == start):
            continue
        event = store.upsert_event(
            known["id"] if known else None,
            {
                "name": entry["name"],
                "start_date": start,
                "date_precision": date_precision(start, None),
                "place": entry["place"],
            },
            people=[pid for name in entry["people"] if (pid := person(name, model_call_id)) is not None],
            model_call_id=model_call_id,
        )
        couple_of(store, event, model_call_id=model_call_id)


def same_kind(store: Store, a: str, b: str) -> bool:
    """Whether a happening named `a` can be the record event named `b`. The model links a happening to a
    wrong record event at times, such as a move to a birthday, and a wrong link moves a date. So the names
    must share a word other than a person's name or a kind of event, such as "Iceland" in "Emily's Iceland
    trip" and "Iceland Trip 2022". Names with no such words must share their kind, such as "birthday". A
    name with no words at all, such as "...", leaves the link to the model."""
    names = {w.lower() for p in store.graph.people.values() for w in p["name"].split()}

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]+", text.lower().replace("'s", "")) if len(w) > 2} - names

    if not words(a):
        return True
    specific_a, specific_b = words(a) - EVENT_KINDS, words(b) - EVENT_KINDS
    if specific_a or specific_b:
        return bool(specific_a & specific_b)
    return bool(words(a) & words(b))


def same_event(name: str, start: str | None, event: Record) -> bool:
    """Whether a listed event is a record event: the same name, or a close name in the same year."""
    if name.lower() == event["name"].lower():
        return True
    same_year = start and event["start_date"] and start[:4] == event["start_date"][:4]
    return bool(same_year) and SequenceMatcher(None, name.lower(), event["name"].lower()).ratio() >= SAME_EVENT
