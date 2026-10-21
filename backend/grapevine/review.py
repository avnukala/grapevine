"""The review pass: the model reads the whole transcript beside the record and completes the relationships,
group members, and events that single turns left out. One turn rarely says that two housemates are friends
or who is on a team. The whole conversation does.

Three calls run at once, one for each kind of record, because one call asked for all of them drops items.
Each call writes short notes before its list, which the list then follows."""

import asyncio
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from grapevine.extract import Ambiguity, Teller, clean_date, date_precision, resolve, year_in
from grapevine.graph import Record
from grapevine.llm import LLM, Call, watch_on
from grapevine.names import misspelled
from grapevine.store import Store, proper_group_name

PROMPT_VERSION = "review-v13"
SAME_EVENT = 0.6
VOTES = 3
VOTE_TEMPERATURE = 0.5

INTRO = """You check the record of a group conversation against its full transcript, and complete it. Return JSON only.

The input lists today's date, the people (the ones in the room are marked), the groups, the events, and the relationships in the record, then the numbered transcript with the speaker of each turn. Refer to people by the full names in the record. Write a new person's name as said.

"""

RELATIONSHIPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: one line for each other pair the conversation ties together, such as a couple whose wedding someone mentions, a sibling, a cousin, an ex, or a coworker. Then fill "room", which has one key for each pair of people in the room, and "other", for the pairs in your analysis.

- For each pair in "room", list its relationships, or an empty list when the conversation does not show how these two know each other. Being at the same party or in the same room is not a tie.
- List every relationship the conversation states or clearly implies, including the ones in the record. Fix a record entry when the transcript shows a better category, status, label, or meeting date.
- Infer what the conversation implies. People who share a house, or hold a house meeting, are roommates and friends. A couple who is married or planning a wedding is romantic, and two people whose wedding someone went to are married. Exes who are still in each other's lives, for example one helps plan the other's wedding, are also friends.
- category:
  - romantic: dating, partners, engaged, married, exes.
  - family: parents, children, siblings, cousins, in-laws.
  - friend: friends, including roommates, bandmates, teammates, climbing partners, and league mates, unless the conversation says they are not friends.
  - professional: coworkers, bosses, managers, employees, clients, business partners.
  - other: only acquaintances with no other tie.
- A pair can have two entries. Exes who stayed friends get romantic with status past and friend with status current.
- status: current or past, for the tie itself. Exes are romantic and past.
- in_touch: true when the two are still in each other's lives today, for example old roommates who still meet, or an ex who helps plan the other's wedding. False when the conversation says they lost touch or one died.
- met_on: when the pair met, as "YYYY" or "YYYY-MM", when the conversation tells or implies it. A couple who met at Bonnaroo in June 2015 has "2015-06". If the conversation only says when the relationship began, such as "we dated from 2020 to 2022", use that. Coworkers met when the second one started the job. Else null.
- label: a 1 to 3 word term, for example "college roommate", "ex-girlfriend", "cousin". summary: one sentence."""
)

GROUPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: for each named group (a crew, a team, a band, a club, a league), the plain phrases people use for it, such as "the crew" or "the league", and each line of the transcript that puts someone in it under any of those names. Then fill "known", which has one key for each group in the record, and "new", for named groups the record does not have.

- For each known group, list its plain phrases in "also_called", and answer for each person whether they are in it under any of its names. People who say "we", "our team", or "our league" about the group are in it, and so are the people they list as going with "the whole crew".
- Every named group in your analysis that is not under "known" goes in "new", with one entry under its proper name, for example "Sunday Sends", and everyone in it.
- A house, a meeting, or an event is not a group."""
)

MEMBERS_SYSTEM = """You read the lines of a group conversation that mention one group, and list who is in it. Return JSON only.

Write "analysis" first: for each line, who it puts in the group. Then list "members" with the full names under PEOPLE when they match.
- The speaker of a line is in the group when they say "we", "our", or "me" about it, or call themselves a member.
- People a line names as going with the whole group are in it, for example "the whole crew is going: me, Owen, and Kevin".
- People a line calls members are in it."""

EVENTS_SYSTEM = (
    INTRO
    + """Write "analysis" first: one line for each event in the record with the date the transcript supports, then one line for each other dated happening in the transcript, such as a trip, a wedding, a party, an accident, a show, a move to a new city, or a move into a house. Then fill "record", which has one key for each event in the record, and "added", for the other happenings.

- start_date: "YYYY", "YYYY-MM", or "YYYY-MM-DD", resolved with today's date when the conversation says "last weekend" or "tonight". When people disagree on a date, use the date of the person who was there, or the date that the reasons given support, for example "Paige got married in 2021, so the trip was 2022".
- Name a new event by what and who, for example "Nina's move to Denver". people: who took part."""
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
    """One required date for each event in the record, so the model checks each one against the transcript."""
    return _object(
        analysis=TEXT,
        record=_object(**{name: _object(start_date=NULLABLE) for name in record}),
        added={"type": "array", "items": _object(name=TEXT, start_date=NULLABLE, place=NULLABLE, people=NAMES)},
    )


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
            entries = agreed(room, person, call_id, needed=len(room) // 2 + 1)
            entries += agreed([o["other"] for o in outputs], person, call_id, needed=1)
            apply_relationships(store, entries, person, call_id)
        if groups.output:
            apply_groups(store, found_groups, person, groups.model_call_id)
        if events.output:
            for name, answer in events.output["record"].items():
                start = clean_date(answer["start_date"])
                event = store.graph.events[record_events[name]]
                if start and start != event["start_date"]:
                    fields = {"start_date": start, "date_precision": date_precision(start, None)}
                    store.upsert_event(event["id"], fields, people=[], model_call_id=events.model_call_id)
            apply_events(store, events.output["added"], person, events.model_call_id)


async def members_of(store: Store, llm: LLM, group: Record, *, model: str, seed: int) -> list[str]:
    """Asks who is in one group, with only the lines that mention it under any of its names. In the whole
    transcript, the model misses a line such as "the whole crew is going: me, Owen, and Kevin"."""
    words = [group["name"], *group["also_called"]]
    keys = {w.lower().removeprefix("the ").removeprefix("our ").removeprefix("my ").strip() for w in words}
    names = {pid: p["name"] for pid, p in store.graph.people.items()}
    turns = sorted(store.graph.segments.values(), key=lambda s: s["start_ms"])
    lines = [
        f"{names.get(s['person_id'], 'Unknown speaker')}: {s['text']}"
        for s in turns
        if any(key and key in s["text"].lower() for key in keys)
    ]
    if not lines:
        return []
    user = "\n".join(
        [f"GROUP: {group['name']} (also called {', '.join(group['also_called']) or 'nothing else'})", "", "PEOPLE:"]
        + [f"- {name}" for name in names.values()]
        + ["", "LINES:", *lines]
    )
    call = Call("reviewer", model, PROMPT_VERSION, MEMBERS_SYSTEM, user, MEMBERS_SCHEMA, seed, max_tokens=1500)
    result = await llm.call(call, run_id=store.run_id, audio_ms=store.audio_ms)
    return result.output["members"] if result.output else []


def agreed(lists: list[list[Record]], person, model_call_id: int, *, needed: int) -> list[Record]:
    """The relationships that at least `needed` votes list, by pair and category. Each vote samples the model
    at a small temperature. Pairs in the room need most votes, because the model is quick to call any two
    guests friends. Other pairs need one vote, because the conversation names them outright."""
    seen: dict[tuple[frozenset, str], list[Record]] = defaultdict(list)
    for entries in lists:
        for entry in {
            (frozenset((entry["a"], entry["b"])), entry["category"]): entry for entry in ties(entries)
        }.values():
            a, b = person(entry["a"], model_call_id), person(entry["b"], model_call_id)
            if a is not None and b is not None and a != b:
                seen[(frozenset((a, b)), entry["category"])].append(entry)
    return [
        {**entries[0], "met_on": next((e["met_on"] for e in entries if e["met_on"]), None)}
        for entries in seen.values()
        if len(entries) >= needed
    ]


def ties(entries: list[Record]) -> list[Record]:
    """The relationships that each listed tie stands for. A tie between people still in touch is current, and
    exes still in touch are also friends."""
    result = []
    for entry in entries:
        if entry["in_touch"] and entry["category"] == "romantic" and entry["status"] == "past":
            result += [entry, {**entry, "category": "friend", "label": "friend", "status": "current", "met_on": None}]
        elif entry["in_touch"]:
            result.append({**entry, "status": "current"})
        else:
            result.append(entry)
    return result


def apply_relationships(store: Store, entries: list[Record], person, model_call_id: int) -> None:
    """Writes each listed relationship. For a pair the review lists, it also removes the categories the review
    left out, because the review read the whole conversation and a single turn did not."""
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
        if categories and relationship["category"] not in categories:
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


def apply_events(store: Store, entries: list[Record], person, model_call_id: int) -> None:
    """Adds each new event, and moves a record event to the date the review gives it."""
    for entry in entries:
        start = clean_date(entry["start_date"]) or year_in(entry["name"])
        known = next((e for e in store.graph.events.values() if same_event(entry["name"], start, e)), None)
        if known and (start is None or known["start_date"] == start):
            continue
        store.upsert_event(
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


def same_event(name: str, start: str | None, event: Record) -> bool:
    """Whether a listed event is a record event: the same name, or a close name in the same year."""
    if name.lower() == event["name"].lower():
        return True
    same_year = start and event["start_date"] and start[:4] == event["start_date"][:4]
    return bool(same_year) and SequenceMatcher(None, name.lower(), event["name"].lower()).ratio() >= SAME_EVENT
