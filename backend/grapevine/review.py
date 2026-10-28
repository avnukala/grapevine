"""The review pass: the model reads the whole transcript beside the record and completes the relationships,
group members, and events that single turns left out. One turn rarely says that two housemates are friends
or who is on a team. The whole conversation does.

Three calls run at once, one for each kind of record, because one call asked for all of them drops items.
Each call writes short notes before its list, which the list then follows."""

import asyncio
import re
from collections import defaultdict
from dataclasses import replace
from difflib import SequenceMatcher

from grapevine.extract import (
    Ambiguity,
    Removed,
    Teller,
    category_of,
    clean_date,
    couple_of,
    date_precision,
    resolve,
    year_in,
)
from grapevine.graph import Record
from grapevine.llm import LLM, Call, Result, watch_on
from grapevine.names import misspelled
from grapevine.store import GROUP_WORDS, Store, group_key, proper_group_name

PROMPT_VERSION = "review-v28"
SAME_EVENT = 0.6
# The prompt says that how two people met, or that they dated, is not a happening, and the model still lists it.
NOT_HAPPENINGS = {"relationship", "meeting", "met", "dating", "friendship"}
EVENT_KINDS = {"the", "and", "with", "trip", "party", "event", "wedding", "birthday", "move", "visit", "weekend"}
VOTES = 3
LINE_MATCH = 0.8
WORD_MATCH = 0.6
MAX_CANDIDATES = 8
THE_NAME = re.compile(r"\b(The [A-Z][\w'-]*(?: [A-Z][\w'-]*)*)")
NOUN_NAME = re.compile(r"\b((?:[A-Z][\w'-]* )+(?:crew|team|league|club|band|Crew|Team|League|Club|Band))\b")
QUOTE_MATCH = 0.8
SPEAKER = re.compile(r"^\s*[A-Z][\w'-]*(?: [A-Z][\w'-]*)*:\s")
VOTE_TEMPERATURE = 0.5
RETRY_TEMPERATURE = 0.3

INTRO = """You check the record of a group conversation against its full transcript, and complete it. Return JSON only.

The input lists today's date, the people (the ones in the room are marked), the groups, the events, and the relationships in the record, then the numbered transcript with the speaker of each turn. Refer to people by the full names in the record. Write a new person's name as said.

"""

RELATIONSHIPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: one line for each pair the conversation ties together. Start with the pairs in the room, such as housemates, a couple, old roommates, or coworkers, then the others, such as a couple whose wedding someone mentions, a sibling, a cousin, or an ex. Include pairs where neither person is in the room: "my cousin's wedding, Tom and Rachel" makes Tom and Rachel a married couple. Then list "ties", one entry for each relationship in your analysis.

- List only pairs that a line ties together. Being at the same dinner or party, in the same conversation, or friends of the same host is not a tie, so most pairs in a big room get no entry.
- List every relationship the conversation states, including the ones in the record. Fix a record entry when the transcript shows a better category, status, label, or meeting date.
- quote: the words of one transcript line that state the tie, copied exactly.
- Some lines imply a tie. People who share a house, or hold a house meeting, are roommates and friends. A couple who is married or planning a wedding is romantic, and two people whose wedding someone went to are married. Exes who are still in each other's lives, for example one still helps the other move house, are also friends.
- category:
  - romantic: dating, partners, engaged, married, exes.
  - family: parents, children, siblings, cousins, in-laws.
  - friend: friends, including roommates, bandmates (also in a band that plays paid gigs), teammates, climbing partners, league mates, and co-founders of a league or club, unless the conversation says they are not friends.
  - professional: coworkers, bosses, managers, employees, clients, business partners.
  - other: only acquaintances with no other tie.
- A pair can have two entries. Exes who stayed friends get romantic with status past and friend with status current.
- status: current or past, for the tie itself. Exes are romantic and past.
- in_touch: true when the two are still in each other's lives today, for example old roommates who still meet, a friend who gave a toast at the other's recent wedding, or an ex who still helps the other move house. False when the conversation says they lost touch or one died.
- met_on: when the pair first met, as "YYYY" or "YYYY-MM", when the conversation tells or implies it. A couple who met at Coachella in April 2016 has "2016-04". If the conversation only says when the relationship began, such as "we dated from 2020 to 2022", use that. Coworkers met when the second one started the job. Never the date of a proposal, an engagement, a wedding, or a move. Else null.
- label: a 1 to 3 word term for how they first knew each other, for example "college roommate", "ex-girlfriend", "cousin", not a later shared hobby such as "league mates" for old roommates.
- summary: one or two sentences on how and where they met, who introduced them, and when the tie changed, for example "Met as freshman roommates in Lisbon in 2012, introduced by Dana Cole. Dated until 2019, then stayed friends." Only what the conversation says."""
)

GROUPS_SYSTEM = (
    INTRO
    + """Write "analysis" first: for each named group (a crew, a team, a band, a club, a league) that anyone names, even once, such as a league someone runs, the plain phrases people use for it, such as "the crew" or "the league", and each line of the transcript that puts someone in it under any of those names. Then fill "known", which has one key for each group in the record, "candidates", which has one key for each name in the transcript that could be a group, and "new", for other named groups the record does not have.

- For each known group, list its plain phrases in "also_called", and answer for each person whether a line puts them in it under any of its names. People who say "we", "our team", or "our league" about the group are in it, and so are the people they list as going with "the whole crew". Asking about a group, or talking near it, does not put someone in it.
- For each candidate, is_group is true only when the name is a group of people, such as a crew, a team, a band, a club, a league, or a friend group, that a line says someone in the transcript belongs to. A trivia team someone reports a result for is a group. A band people only went to see, an event, a place, or a misheard phrase is not.
- Every other named group in your analysis that is not under "known" or "candidates" goes in "new", with one entry under its proper name, spelled as its members would write it, for example "Thursday Flyers" when the transcript says "Thursday Fliers", and everyone in it.
- A house, a meeting, an event, or a regular activity such as "poker nights" is not a group."""
)

MEMBERS_SYSTEM = """You read the lines of a group conversation that mention one group, and list who is in it. Return JSON only.

Write "analysis" first: for each line, who it puts in the group. Some lines are about something else and put no one in it. Then list "members" with the full names under PEOPLE when they match.
- The speaker of a line is in the group when they say "we", "our", or "me" about this group, or call themselves a member. A "we" in a story or about another group does not count.
- The speaker of a line is in the group when they talk about this group from the inside, such as how it is run or how it did, even without naming it or saying "we". "Which she runs like a tyrant", said about the club's president, puts the speaker in the club. "The team lost again, and I am livid" puts the speaker in the team.
- Asking whether someone goes with the group, or telling who leads it, does not by itself put the speaker in it.
- People a line names as going with the whole group are in it, for example "the whole crew is going: me, Dev, and Lena".
- A line can answer the line before it, for example "Are you going with the Thursday Flyers crew?" "Yes, me, Dev, and Lena."
- "You" in a line is the person who answers it, not the speaker before it. "So you signed up for the league?" puts the next speaker in it.
- People a line calls members are in it. Speech recognition misspells group names, so "Thursday Fliers" is "Thursday Flyers".
- Speech recognition also mishears first names. A name in a line that sounds like exactly one person under PEOPLE, such as "Erin" for "Aaron Cole", is that person."""

EVENTS_SYSTEM = (
    INTRO
    + """Write "analysis" first: go through the transcript in order, and write one line for every happening that the conversation gives a time for (a year, a season, "last weekend"), whether or not the record has it. Look for trips, weddings (also one said in passing, such as "Rosa got married in 2021, so it couldn't be then"), engagements, parties, accidents, injuries, shows, moves to a new city, and moves into a house. A time can come in the next line, for example "Greta left me for Seattle." "In 2017." Keep each line under 25 words, and do not argue with yourself. Then list each one in "happenings".

- List every happening of your analysis, including one whose date people dispute.
- Not happenings: how or when two people met, a relationship, a job, school years, a visit planned for the coming days, or a game result.
- A show, a party, or a game that people go to today or tonight is a happening, dated today.
- record: the record event that is the same happening, or null. A move is never a birthday or a visit.
- start_date: only as precise as what was said. "YYYY-MM-DD" only for a named day or a day near today, such as "last Saturday" or "tonight", resolved with today's date. "YYYY-MM" only for a named month or "last month". A season, "years ago", or a year alone is "YYYY". Null when no year can be worked out. When people disagree on a date, use the date of the person who was there, or the date that the reasons given support, for example "Rosa got married in 2021, so the trip was 2022".
- name: what and who, for example "Greta's move to Seattle" or "Rosa's wedding". people: who took part."""
)

SMALL_TALK_SYSTEM = """You read the facts recorded from a group conversation beside its transcript, and mark the ones that are not lasting history about the people. Return JSON only.

For each fact, write a short note, then "lasting".
- lasting is false for small talk: a remark about tonight or this gathering (the food, the game being played, who drives, who is late, a reservation), an errand or chore (a car inspection, who does the dishes, cooking on Saturday), a mood or wish of the moment, a joke or tease, a question, or a request to hear a story.
- lasting is true for history: jobs, homes (also a town someone drives in from), house rules, traditions (also ones people plan to start), moves, school, family, pets, hobbies, groups, relationships, how people met, life events such as an engagement, a wedding, an injury, or a trip, a story from the past, and plans to host or go to a party, a holiday, a trip, or a wedding.
- When unsure, lasting is true."""

NULLABLE = {"type": ["string", "null"]}
NAMES = {"type": "array", "items": {"type": "string"}}
TEXT = {"type": "string"}
# At temperature 0 the model can repeat one sentence in its notes until it runs out of tokens.
ANALYSIS = {"type": "string", "maxLength": 1500}


def _object(**properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


TIE = {
    "quote": TEXT,
    "category": {"type": "string", "enum": ["romantic", "family", "friend", "professional", "other"]},
    "label": TEXT,
    "status": {"type": "string", "enum": ["current", "past"]},
    "met_on": NULLABLE,
    "in_touch": {"type": "boolean"},
    "summary": TEXT,
}


RELATIONSHIPS_SCHEMA = _object(analysis=ANALYSIS, ties={"type": "array", "items": _object(a=TEXT, b=TEXT, **TIE)})


def groups_schema(groups: list[str], candidates: list[str], people: list[str]) -> dict:
    """One required answer for each person in each known group, so the model decides about everyone, and one
    for each candidate name, so the model does not skip a group that no turn recorded."""
    members = _object(**{name: {"type": "boolean"} for name in people})
    candidate = _object(is_group={"type": "boolean"}, also_called=NAMES, members=members)
    return _object(
        analysis=ANALYSIS,
        known=_object(**{group: _object(also_called=NAMES, members=members) for group in groups}),
        candidates=_object(**{name: candidate for name in candidates}),
        new={"type": "array", "items": _object(name=TEXT, also_called=NAMES, members=NAMES)},
    )


def candidate_groups(texts: list[str], known: list[str]) -> list[str]:
    """Names in a transcript that could be a group, such as "The No-It-Alls" or "Sunday Sense crew", other
    than the known groups. A turn can report a group's result without the model recording the group."""
    found = [m[1] for text in texts for pattern in (THE_NAME, NOUN_NAME) for m in pattern.finditer(text)]
    keys = [group_key(name) for name in known]
    candidates: list[str] = []
    for name in dict.fromkeys(found):
        key = group_key(name)
        if proper_group_name(name) and not any(SequenceMatcher(None, key, k).ratio() >= LINE_MATCH for k in keys):
            candidates.append(name)
            keys.append(key)
    return candidates[:MAX_CANDIDATES]


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
    return _object(analysis=ANALYSIS, happenings={"type": "array", "items": happening})


MEMBERS_SCHEMA = _object(analysis=ANALYSIS, members=NAMES)
NOTE = {"type": "string", "maxLength": 100}


def small_talk_message(store: Store) -> tuple[str, dict, dict[str, int]]:
    """The personal facts and stories that turns recorded, each under its ref, then the transcript. Ties,
    corrections, and the evidence of a relationship are history, and the model called a couple small talk."""
    graph = store.graph
    names = {pid: p["name"] for pid, p in graph.people.items()}
    evidence = {fact_id for r in graph.relationships.values() for fact_id in r["evidence"]}
    facts = {
        graph.ref("f", f["id"]): f
        for f in graph.live_facts()
        if f["segment_ids"] and f["kind"] in ("personal", "story") and f["id"] not in evidence
    }
    turns = sorted(graph.segments.values(), key=lambda s: s["start_ms"])
    lines = ["FACTS:", *(f"{ref} {f['text']}" for ref, f in facts.items()), "", "TRANSCRIPT:"]
    lines += [f"{names.get(s['person_id'], 'Unknown speaker')}: {s['text']}" for s in turns]
    schema = _object(**{ref: _object(note=NOTE, lasting={"type": "boolean"}) for ref in facts})
    return "\n".join(lines), schema, {ref: f["id"] for ref, f in facts.items()}


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
    people = list(dict.fromkeys(p["name"] for p in names.values() if p["merged_into_id"] is None))
    record_events = {e["name"]: e["id"] for e in reversed(list(store.graph.events.values()))}

    def ask(system: str, schema: dict, *, seed: int = seed, temperature: float = 0) -> Call:
        return Call("reviewer", model, PROMPT_VERSION, system, user, schema, seed, 6000, temperature)

    watch = watch_on(store, worker="reviewer")
    votes = [
        ask(RELATIONSHIPS_SYSTEM, RELATIONSHIPS_SCHEMA, seed=seed + i, temperature=VOTE_TEMPERATURE)
        for i in range(VOTES)
    ]
    known_groups = [g["name"] for g in store.graph.groups.values()]
    turns = sorted(store.graph.segments.values(), key=lambda s: s["start_ms"])
    texts = [s["text"] for s in turns]
    groups_call = ask(GROUPS_SYSTEM, groups_schema(known_groups, candidate_groups(texts, known_groups), people))
    small_talk_user, small_talk_schema, small_talk_facts = small_talk_message(store)
    small_talk_call = Call(
        "reviewer", model, PROMPT_VERSION, SMALL_TALK_SYSTEM, small_talk_user, small_talk_schema, seed, 6000
    )
    groups, events, small_talk, *voted = await asyncio.gather(
        call_or_retry(store, llm, groups_call),
        call_or_retry(store, llm, ask(EVENTS_SYSTEM, events_schema(list(record_events)))),
        call_or_retry(store, llm, small_talk_call),
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
            for name, answer in [
                *groups.output["known"].items(),
                *((n, a) for n, a in groups.output["candidates"].items() if a["is_group"]),
            ]
        ] + groups.output["new"]
        more = await asyncio.gather(*[members_of(store, llm, g, model=model, seed=seed) for g in found_groups])
        found_groups = [
            {**g, "members": [*extra, *said_in_lines(g, texts)], "named": [*g["members"], *extra]}
            for g, extra in zip(found_groups, more)
        ]

    def person(name: str, model_call_id: int) -> int | None:
        known = [p for p in store.graph.people.values() if misspelled(p["name"], name)]
        if len(known) == 1 and name not in {p["name"] for p in store.graph.people.values()}:
            return known[0]["id"]
        teller = Teller(segment_id=-1, person_id=None, confidence=1.0, model_call_id=model_call_id)
        found = resolve(store, name, None, teller=teller, answers={}, text="")
        return None if isinstance(found, Ambiguity | Removed) else found

    with store.batch():
        outputs = [v.output for v in voted if v.output]
        if outputs:
            call_id = voted[0].model_call_id
            stated_ties = [[tie for tie in output["ties"] if quoted(tie["quote"], texts)] for output in outputs]
            kept = agreed(stated_ties, person, call_id, needed=len(outputs) // 2 + 1)
            room = set(store.graph.room())
            in_room = [t for t in kept if {person(t["a"], call_id), person(t["b"], call_id)} <= room]
            apply_relationships(store, in_room, person, call_id, prune=True)
            apply_relationships(store, [t for t in kept if t not in in_room], person, call_id, prune=False)
        if events.output:
            apply_happenings(store, events.output["happenings"], record_events, person, events.model_call_id)
        if groups.output:
            apply_groups(store, found_groups, person, groups.model_call_id)
        if small_talk.output:
            hide_small_talk(store, small_talk.output, small_talk_facts, small_talk.model_call_id)


def quoted(quote: str, texts: list[str]) -> bool:
    """Whether each sentence of a quote is in a transcript line, allowing for small slips in the copy and a
    speaker name before it. The model joins sentences from several lines into one quote. A tie the model
    cannot quote, such as two guests at one dinner being "friends", is one it made up."""
    lines = [re.findall(r"[a-z0-9']+", text.lower()) for text in texts]

    def found(sentence: str) -> bool:
        wanted = re.findall(r"[a-z0-9']+", SPEAKER.sub("", sentence).lower())
        return any(
            sum(b.size for b in SequenceMatcher(None, wanted, line, autojunk=False).get_matching_blocks())
            >= QUOTE_MATCH * len(wanted)
            for line in lines
        )

    sentences = [sentence for sentence in re.split(r"[.!?…]+", quote) if re.search(r"\w", sentence)]
    return bool(sentences) and all(found(sentence) for sentence in sentences)


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
    result = await call_or_retry(store, llm, call)
    return result.output["members"] if result.output else []


async def call_or_retry(store: Store, llm: LLM, call: Call) -> Result:
    """Asks once more with another seed and a small temperature when the output is not valid JSON, such as
    JSON cut off when the model loops until it runs out of tokens."""
    result = await llm.call(call, run_id=store.run_id, audio_ms=store.audio_ms)
    if result.output is not None:
        return result
    retry = replace(call, seed=call.seed + VOTES, temperature=RETRY_TEMPERATURE)
    return await llm.call(retry, run_id=store.run_id, audio_ms=store.audio_ms)


def group_lines(texts: list[str], group_names: list[str]) -> list[int]:
    """The indexes of the lines about a group: lines that name it under any spelling close to one of its
    names, the line before each, which tells who "you" is, the two lines after each, which often answer it,
    and lines that use its plain noun, such as "the whole crew", with the line after each. Each word of a name
    must be close to the word said, so "now it's five" does not name "Know It Alls"."""

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    keys = [words(group_key(name)) for name in group_names if group_key(name)]

    def close(said: list[str], key: list[str]) -> bool:
        return SequenceMatcher(None, " ".join(said), " ".join(key)).ratio() >= LINE_MATCH and all(
            SequenceMatcher(None, a, b).ratio() >= WORD_MATCH for a, b in zip(said, key)
        )

    def names_it(text: str) -> bool:
        said = words(text)
        return any(close(said[i : i + len(key)], key) for key in keys for i in range(len(said) - len(key) + 1))

    named = [i for i, text in enumerate(texts) if names_it(text)]
    near = {j for i in named for j in range(max(i - 1, 0), min(i + 3, len(texts)))}
    nouns = {w for i in near for w in words(texts[i]) if w in GROUP_WORDS} | {
        w for name in group_names for w in words(name) if w in GROUP_WORDS
    }
    plain = [i for i, text in enumerate(texts) if nouns & set(words(text))]
    return sorted(near | {j for i in plain for j in range(i, min(i + 2, len(texts)))})


def said_in_lines(group: Record, texts: list[str]) -> list[str]:
    """The members the whole-transcript call gives that the lines about the group name, or all of them when
    no line is about it. That call puts the speaker of a line about a league in it, such as "Arjun, you're
    in the league", while it also finds members that the members call misses."""
    picked = group_lines(texts, [group["name"], *group["also_called"]])
    said = {w for i in picked for w in re.findall(r"[a-z']+", texts[i].lower())}
    return [name for name in group["members"] if not picked or name.split()[0].lower() in said]


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
    Only pairs in the room prune. The review hears both of them, while it can miss a category that a turn got
    right about people who are not there."""
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
    """Adds the members that the lines about a group name, or that the whole transcript names when no line
    is about it. The whole transcript puts a guest in a league that a line beside him lists. A record member
    that neither call names leaves the group, and a group that is a record event, such as "Friendsgiving",
    loses all its members."""
    events = {group_key(re.sub(r"\d+", "", e["name"]).strip()) for e in store.graph.events.values()}
    for group in list(store.graph.groups.values()):
        if group_key(group["name"]) in events:
            for pid in {m["person_id"] for m in store.graph.members.values() if m["group_id"] == group["id"]}:
                store.remove_member(group_id=group["id"], person_id=pid)
    for entry in [e for e in entries if proper_group_name(e["name"]) and group_key(e["name"]) not in events]:
        found = [pid for name in entry["members"] if (pid := person(name, model_call_id)) is not None]
        if not found:
            continue
        group = store.group_for(entry["name"])
        known = {m["person_id"] for m in store.graph.members.values() if m["group_id"] == group["id"]}
        named = {person(name, model_call_id) for name in entry["named"]}
        for pid in known - named:
            store.remove_member(group_id=group["id"], person_id=pid)
        new = list(dict.fromkeys(pid for pid in found if pid not in known))
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


def hide_small_talk(store: Store, answers: Record, facts: dict[str, int], model_call_id: int) -> None:
    """Hides each fact the review calls small talk the way a correction hides a fact, so the change log keeps
    it. A fact revised by itself has no newer version."""
    for ref, answer in answers.items():
        if not answer["lasting"]:
            store.update_fact(facts[ref], revised_by_fact_id=facts[ref], model_call_id=model_call_id)


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
