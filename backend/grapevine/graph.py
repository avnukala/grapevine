"""The in-memory graph of one run, rebuilt from its change ops.

Each op carries the full record it writes, so applying an op replaces the record. The client
applies the same ops with the same rules in `src/state/reducer.ts`.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal, assert_never

Phase = Literal["intro", "conversation", "review", "ended"]
RefKind = Literal["p", "e", "f", "r"]
Record = dict[str, Any]


@dataclass
class Graph:
    people: dict[int, Record] = field(default_factory=dict)
    speakers: dict[str, int | None] = field(default_factory=dict)
    segments: dict[int, Record] = field(default_factory=dict)
    facts: dict[int, Record] = field(default_factory=dict)
    events: dict[int, Record] = field(default_factory=dict)
    groups: dict[int, Record] = field(default_factory=dict)
    members: dict[tuple[int, int, int | None], Record] = field(default_factory=dict)
    relationships: dict[int, Record] = field(default_factory=dict)
    prompts: dict[int, Record] = field(default_factory=dict)
    roster: list[int] = field(default_factory=list)
    removed: set[int] = field(default_factory=set)
    phase: Phase = "intro"

    def apply(self, op: Record) -> None:
        kind = op["type"]
        match kind:
            case "person":
                self.people[op["person"]["id"]] = op["person"]
                self.removed.discard(op["person"]["id"])
            case "person_removed":
                self._remove_person(op["id"])
            case "speaker":
                self.speakers[op["speaker"]["label"]] = op["speaker"]["person_id"]
            case "segment":
                self.segments[op["segment"]["id"]] = op["segment"]
            case "fact":
                self.facts[op["fact"]["id"]] = op["fact"]
            case "event":
                self.events[op["event"]["id"]] = op["event"]
            case "group":
                self.groups[op["group"]["id"]] = op["group"]
            case "member":
                member = op["member"]
                self.members[(member["group_id"], member["person_id"], member["source_fact_id"])] = member
            case "relationship":
                self.relationships[op["relationship"]["id"]] = op["relationship"]
            case "relationship_removed":
                self.relationships.pop(op["id"], None)
            case "prompt":
                self.prompts[op["prompt"]["id"]] = op["prompt"]
            case "phase":
                self.phase = op["phase"]
            case "roster":
                self.roster = op["person_ids"]
            case _:
                assert_never(kind)

    def _remove_person(self, person_id: int) -> None:
        """Drops a person the room deleted, with every fact, membership, and event part that names them."""
        self.people.pop(person_id, None)
        self.removed.add(person_id)
        self.roster = [p for p in self.roster if p != person_id]
        self.members = {key: m for key, m in self.members.items() if m["person_id"] != person_id}
        self.facts = {fid: f for fid, f in self.facts.items() if all(p["person_id"] != person_id for p in f["people"])}
        for event_id, event in self.events.items():
            if any(p["person_id"] == person_id for p in event["people"]):
                people = [p for p in event["people"] if p["person_id"] != person_id]
                self.events[event_id] = {**event, "people": people}

    def room(self) -> list[int]:
        """The people in the room: everyone on the roster, and anyone a voice belongs to."""
        return list(dict.fromkeys([*self.roster, *(p for p in self.speakers.values() if p is not None)]))

    def live_facts(self) -> list[Record]:
        """Facts that no later fact revised."""
        return [f for f in self.facts.values() if f["revised_by_fact_id"] is None]

    def active_prompts(self) -> list[Record]:
        return [p for p in self.prompts.values() if p["dismissed_at_ms"] is None]

    def _records(self, kind: RefKind) -> dict[int, Record]:
        match kind:
            case "p":
                return self.people
            case "e":
                return self.events
            case "f":
                return self.facts
            case "r":
                return self.relationships
            case _:
                assert_never(kind)

    def ref(self, kind: RefKind, record_id: int) -> str:
        """Numbers records 1, 2, 3 in the order the run first wrote them. Model prompts use these refs, so an
        unchanged replay sends the same text although its database ids differ."""
        return f"{kind}{list(self._records(kind)).index(record_id) + 1}"

    def lookup(self, kind: RefKind, ref: str | None) -> int | None:
        match = re.match(rf"{kind}(\d+)", ref or "")
        ids = list(self._records(kind))
        if match is None or not 0 < int(match[1]) <= len(ids):
            return None
        return ids[int(match[1]) - 1]
