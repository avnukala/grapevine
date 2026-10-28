"""Writes the records of one run. Each write also logs a change op, updates the in-memory graph, and
broadcasts the op to connected clients after the transaction commits."""

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from grapevine.graph import Graph, Phase, Record

Broadcast = Callable[[Record], None]

FACT_FIELDS = (
    "kind",
    "text",
    "detail",
    "told_by_person_id",
    "when_text",
    "start_date",
    "end_date",
    "date_precision",
    "event_id",
    "confidence",
    "disputed",
    "revised_by_fact_id",
)
EVENT_FIELDS = ("name", "when_text", "start_date", "end_date", "date_precision", "place")
GROUP_WORDS = {"crew", "team", "group", "club", "league", "gang", "squad", "band"}
PLACES = {"house", "home", "apartment", "street", "meeting"}
PLAIN_WORDS = GROUP_WORDS | {
    "the",
    "our",
    "my",
    "your",
    "their",
    "a",
    "whole",
    "old",
    "new",
    "friend",
    "friends",
    "family",
    "house",
    "meeting",
    "book",
    "trivia",
    "climbing",
    "running",
    "fantasy",
    "football",
    "chat",
    "work",
    "college",
}
SAME_GROUP = 0.85
RELATIONSHIP_FIELDS = ("label", "summary", "status", "met_on", "met_event_id", "closeness", "recency", "sentiment")


def person_record(row: sqlite3.Row) -> Record:
    return {
        "id": row["id"],
        "name": row["name"],
        "aliases": json.loads(row["aliases"]),
        "gender": row["gender"],
        "merged_into_id": row["merged_into_id"],
    }


def group_key(name: str) -> str:
    """A group name without case, a leading "the", or a trailing word that only says it is a group."""
    words = name.lower().split()
    if words[:1] == ["the"]:
        words = words[1:]
    while len(words) > 1 and words[-1] in GROUP_WORDS:
        words = words[:-1]
    return " ".join(words)


def proper_group_name(name: str) -> bool:
    """Whether a group name names one group, such as "Sunday Sends" or "second chapter", and is not a plain
    phrase like "the crew" or "our trivia team", or a place like "the house on Pearl Street". Case does not
    count, because transcripts lose it."""
    words = re.findall(r"[a-z']+", name.lower())
    return not PLACES & set(words) and any(word not in PLAIN_WORDS for word in words)


def name_matches(conn: sqlite3.Connection, name: str) -> list[Record]:
    """People whose full name or alias equals `name`, else whose first name equals a one-word `name`."""
    wanted = name.strip().lower()
    people = [person_record(r) for r in conn.execute("SELECT * FROM people WHERE merged_into_id IS NULL ORDER BY id")]
    exact = [p for p in people if wanted in {p["name"].lower(), *(a.lower() for a in p["aliases"])}]
    if exact or " " in wanted:
        return one_per_full_name(exact)
    return one_per_full_name([p for p in people if p["name"].lower().split()[0] == wanted])


def one_per_full_name(people: list[Record]) -> list[Record]:
    """Two people with the same full name, such as a respelling that copied a known person, are one person to
    the room, so the first stands for both. Asking "Ravi Kapoor or Ravi Kapoor?" has no answer."""
    seen: set[str] = set()
    kept = []
    for person in people:
        key = person["name"].lower()
        if " " in key and key in seen:
            continue
        seen.add(key)
        kept.append(person)
    return kept


class Store:
    def __init__(self, conn: sqlite3.Connection, run_id: int, broadcast: Broadcast | None = None) -> None:
        self.conn = conn
        self.run_id = run_id
        self.graph = Graph()
        self.audio_ms = 0
        self.seq = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM changes WHERE run_id = ?", (run_id,)).fetchone()[0]
        self._broadcast = broadcast or (lambda message: None)
        self._outbox: list[Record] = []
        self._depth = 0

    @contextmanager
    def batch(self) -> Iterator[None]:
        """Groups writes in one transaction. Nested batches join the outer one."""
        self._depth += 1
        if self._depth == 1:
            self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.execute("ROLLBACK")
                self._outbox.clear()
            raise
        self._depth -= 1
        if self._depth == 0:
            self.conn.execute("COMMIT")
            outbox, self._outbox = self._outbox, []
            for message in outbox:
                self._broadcast(message)

    def emit(self, op: Record, *, segment_id: int | None = None, model_call_id: int | None = None) -> None:
        with self.batch():
            self.seq += 1
            self.conn.execute(
                "INSERT INTO changes (run_id, seq, audio_ms, segment_id, model_call_id, op_json) VALUES (?, ?, ?, ?, ?, ?)",
                (self.run_id, self.seq, self.audio_ms, segment_id, model_call_id, json.dumps(op)),
            )
            self.graph.apply(op)
            change = {
                "seq": self.seq,
                "audio_ms": self.audio_ms,
                "segment_id": segment_id,
                "model_call_id": model_call_id,
                "op": op,
            }
            self._outbox.append({"type": "change", "run_id": self.run_id, "change": change})

    def notify(self, message: Record) -> None:
        """Sends a message that is not part of the record, such as who is talking now."""
        self._broadcast({**message, "run_id": self.run_id})

    def set_phase(self, phase: Phase) -> None:
        self.emit({"type": "phase", "phase": phase})

    def person(self, person_id: int) -> Record:
        return person_record(self.conn.execute("SELECT * FROM people WHERE id = ?", (person_id,)).fetchone())

    def show_person(self, person_id: int) -> None:
        """Adds a global person to this run's graph."""
        if person_id not in self.graph.people:
            self.emit({"type": "person", "person": self.person(person_id)})

    def create_person(self, *, name: str, aliases: list[str], gender: str) -> Record:
        with self.batch():
            cursor = self.conn.execute(
                "INSERT INTO people (name, aliases, gender) VALUES (?, ?, ?)", (name, json.dumps(aliases), gender)
            )
            person = self.person(cursor.lastrowid)
            self.emit({"type": "person", "person": person})
        return person

    def name_taken(self, name: str, person_id: int) -> bool:
        """Whether another person, of any session, already has this full name."""
        row = self.conn.execute(
            "SELECT 1 FROM people WHERE lower(name) = ? AND id != ? AND merged_into_id IS NULL",
            (name.lower(), person_id),
        ).fetchone()
        return row is not None

    def update_person(
        self, person_id: int, *, add_aliases: list[str], gender: str | None, name: str | None = None
    ) -> None:
        person = self.person(person_id)
        if name and name.lower() != person["name"].lower() and self.name_taken(name, person_id):
            # Taking the name of another person makes two people that no name can tell apart.
            add_aliases, name = [*add_aliases, name], None
        name = name or person["name"]
        known = {a.lower() for a in person["aliases"]} | {name.lower()}
        aliases = person["aliases"] + [a for a in dict.fromkeys(add_aliases) if a.lower() not in known]
        gender = gender if gender and gender != "unknown" else person["gender"]
        unchanged = (name, aliases, gender) == (person["name"], person["aliases"], person["gender"])
        if unchanged and person_id in self.graph.people:
            return
        with self.batch():
            self.conn.execute(
                "UPDATE people SET name = ?, aliases = ?, gender = ? WHERE id = ?",
                (name, json.dumps(aliases), gender, person_id),
            )
            self.emit({"type": "person", "person": self.person(person_id)})

    def remove_person(self, person_id: int) -> None:
        """Deletes a person from this run. The person stays in the global people table for other runs."""
        with self.batch():
            for rel in [r for r in self.graph.relationships.values() if person_id in (r["a"], r["b"])]:
                self.remove_relationship(rel["id"])
            for label in [lbl for lbl, pid in self.graph.speakers.items() if pid == person_id]:
                self.set_speaker(label, None, None)
            for segment in [s for s in self.graph.segments.values() if s["person_id"] == person_id]:
                self.conn.execute("UPDATE segments SET person_id = NULL WHERE id = ?", (segment["id"],))
                self.emit({"type": "segment", "segment": self.segment(segment["id"])}, segment_id=segment["id"])
            for prompt in [p for p in self.graph.active_prompts() if person_id in p["person_ids"]]:
                self.dismiss_prompt(prompt["id"])
            self.emit({"type": "person_removed", "id": person_id})

    def set_speaker(self, label: str, person_id: int | None, embedding: np.ndarray | None) -> None:
        blob = None if embedding is None else embedding.astype(np.float32).tobytes()
        with self.batch():
            self.conn.execute(
                "INSERT INTO speakers (run_id, speaker_label, person_id, voice_embedding) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (run_id, speaker_label) DO UPDATE SET person_id = excluded.person_id, "
                "voice_embedding = COALESCE(excluded.voice_embedding, voice_embedding)",
                (self.run_id, label, person_id, blob),
            )
            if person_id is not None:
                self.show_person(person_id)
            if self.graph.speakers.get(label, -1) != person_id:
                self.emit({"type": "speaker", "speaker": {"label": label, "person_id": person_id}})

    def voices(self) -> dict[str, tuple[int | None, np.ndarray]]:
        rows = self.conn.execute(
            "SELECT speaker_label, person_id, voice_embedding FROM speakers "
            "WHERE run_id = ? AND voice_embedding IS NOT NULL",
            (self.run_id,),
        )
        return {r[0]: (r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows}

    def segment(self, segment_id: int) -> Record:
        row = self.conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
        return {
            "id": row["id"],
            "start_ms": row["start_ms"],
            "end_ms": row["end_ms"],
            "speaker_label": row["speaker_label"],
            "person_id": row["person_id"],
            "speaker_confidence": row["speaker_confidence"],
            "speaker_manual": bool(row["speaker_manual"]),
            "text": row["text"],
            "timings": json.loads(row["timings_json"]),
        }

    def add_segment(
        self,
        *,
        start_ms: int,
        end_ms: int,
        speaker_label: str,
        person_id: int | None,
        confidence: float | None,
        text: str,
        timings: dict[str, int],
    ) -> Record:
        with self.batch():
            cursor = self.conn.execute(
                "INSERT INTO segments (run_id, start_ms, end_ms, speaker_label, person_id, speaker_confidence, text, "
                "timings_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, start_ms, end_ms, speaker_label, person_id, confidence, text, json.dumps(timings)),
            )
            segment = self.segment(cursor.lastrowid)
            self.emit({"type": "segment", "segment": segment}, segment_id=segment["id"])
        return segment

    def update_segment(
        self,
        segment_id: int,
        *,
        person_id: int | None = None,
        speaker_label: str | None = None,
        speaker_confidence: float | None = None,
        speaker_manual: bool | None = None,
        timings: dict[str, int] | None = None,
        text: str | None = None,
    ) -> None:
        segment = self.segment(segment_id)
        with self.batch():
            self.conn.execute(
                "UPDATE segments SET person_id = ?, speaker_label = ?, speaker_confidence = ?, speaker_manual = ?, "
                "timings_json = ?, text = ? WHERE id = ?",
                (
                    person_id if person_id is not None else segment["person_id"],
                    speaker_label or segment["speaker_label"],
                    speaker_confidence if speaker_confidence is not None else segment["speaker_confidence"],
                    speaker_manual if speaker_manual is not None else segment["speaker_manual"],
                    json.dumps({**segment["timings"], **(timings or {})}),
                    text or segment["text"],
                    segment_id,
                ),
            )
            self.emit({"type": "segment", "segment": self.segment(segment_id)}, segment_id=segment_id)

    def fact(self, fact_id: int) -> Record:
        row = self.conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        people = self.conn.execute("SELECT person_id, role FROM fact_people WHERE fact_id = ?", (fact_id,))
        segments = self.conn.execute("SELECT segment_id FROM fact_segments WHERE fact_id = ?", (fact_id,))
        return {
            "id": row["id"],
            **{f: row[f] for f in FACT_FIELDS},
            "disputed": bool(row["disputed"]),
            "people": [{"person_id": p[0], "role": p[1]} for p in people],
            "segment_ids": [s[0] for s in segments],
        }

    def add_fact(
        self, fields: Record, *, people: list[tuple[int, str]], segment_ids: list[int], model_call_id: int | None
    ) -> Record:
        values = {f: fields.get(f) for f in FACT_FIELDS}
        values["confidence"] = fields.get("confidence", 1.0)
        values["disputed"] = int(fields.get("disputed", False))
        with self.batch():
            cursor = self.conn.execute(
                f"INSERT INTO facts (run_id, {', '.join(FACT_FIELDS)}) VALUES (?, {', '.join('?' * len(FACT_FIELDS))})",
                (self.run_id, *values.values()),
            )
            fact_id = cursor.lastrowid
            for person_id, role in dict(people).items():
                self.show_person(person_id)
                self.conn.execute(
                    "INSERT INTO fact_people (fact_id, person_id, role) VALUES (?, ?, ?)", (fact_id, person_id, role)
                )
            for segment_id in segment_ids:
                self.conn.execute(
                    "INSERT INTO fact_segments (fact_id, segment_id) VALUES (?, ?)", (fact_id, segment_id)
                )
            fact = self.fact(fact_id)
            self.emit(
                {"type": "fact", "fact": fact},
                segment_id=segment_ids[0] if segment_ids else None,
                model_call_id=model_call_id,
            )
        return fact

    def update_fact(self, fact_id: int, *, model_call_id: int | None = None, **fields: Any) -> None:
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self.batch():
            self.conn.execute(f"UPDATE facts SET {assignments} WHERE id = ?", (*fields.values(), fact_id))
            self.emit({"type": "fact", "fact": self.fact(fact_id)}, model_call_id=model_call_id)

    def event(self, event_id: int) -> Record:
        row = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        people = self.conn.execute("SELECT person_id, role FROM event_people WHERE event_id = ?", (event_id,))
        return {
            "id": row["id"],
            **{f: row[f] for f in EVENT_FIELDS},
            "people": [{"person_id": p[0], "role": p[1]} for p in people],
        }

    def upsert_event(
        self, event_id: int | None, fields: Record, *, people: list[int], model_call_id: int | None
    ) -> Record:
        with self.batch():
            if event_id is None:
                cursor = self.conn.execute(
                    f"INSERT INTO events (run_id, {', '.join(EVENT_FIELDS)}) VALUES (?, {', '.join('?' * len(EVENT_FIELDS))})",
                    (self.run_id, *(fields.get(f) for f in EVENT_FIELDS)),
                )
                event_id = cursor.lastrowid
            else:
                updates = {f: fields[f] for f in EVENT_FIELDS if fields.get(f) is not None}
                if updates:
                    self.conn.execute(
                        f"UPDATE events SET {', '.join(f'{k} = ?' for k in updates)} WHERE id = ?",
                        (*updates.values(), event_id),
                    )
            for person_id in people:
                self.show_person(person_id)
                self.conn.execute(
                    "INSERT OR IGNORE INTO event_people (event_id, person_id, role) VALUES (?, ?, 'attendee')",
                    (event_id, person_id),
                )
            event = self.event(event_id)
            if self.graph.events.get(event_id) != event:
                self.emit({"type": "event", "event": event}, model_call_id=model_call_id)
        return event

    def group_for(self, name: str) -> Record:
        """Finds a global group by name or alias, or creates it. Names match loosely, so "Sunday Sends crew"
        and "Sunday Sands" find Sunday Sends."""
        wanted = group_key(name)
        for row in self.conn.execute("SELECT * FROM groups"):
            keys = {group_key(n) for n in [row["name"], *json.loads(row["aliases"])]}
            if any(SequenceMatcher(None, wanted, key).ratio() >= SAME_GROUP for key in keys):
                group = {
                    "id": row["id"],
                    "name": row["name"],
                    "aliases": json.loads(row["aliases"]),
                    "description": row["description"],
                }
                break
        else:
            cursor = self.conn.execute("INSERT INTO groups (name) VALUES (?)", (name.strip(),))
            group = {"id": cursor.lastrowid, "name": name.strip(), "aliases": [], "description": None}
        if group["id"] not in self.graph.groups:
            self.emit({"type": "group", "group": group})
        return group

    def add_member(self, *, group_id: int, person_id: int, since: str | None, until: str | None, fact_id: int) -> None:
        """Adds a person to a group once. A second fact that names the same member adds nothing."""
        if any(key[:2] == (group_id, person_id) for key in self.graph.members):
            return
        member = {
            "group_id": group_id,
            "person_id": person_id,
            "since": since,
            "until": until,
            "source_fact_id": fact_id,
        }
        with self.batch():
            self.show_person(person_id)
            self.conn.execute(
                "INSERT OR IGNORE INTO group_members (group_id, person_id, since, until, source_fact_id) "
                "VALUES (?, ?, ?, ?, ?)",
                tuple(member.values()),
            )
            self.emit({"type": "member", "member": member})

    def relationship(self, relationship_id: int) -> Record:
        row = self.conn.execute("SELECT * FROM relationships WHERE id = ?", (relationship_id,)).fetchone()
        evidence = self.conn.execute(
            "SELECT fact_id FROM relationship_evidence WHERE relationship_id = ? ORDER BY fact_id", (relationship_id,)
        )
        return {
            "id": row["id"],
            "a": row["person_a_id"],
            "b": row["person_b_id"],
            "category": row["category"],
            **{f: row[f] for f in RELATIONSHIP_FIELDS},
            "evidence": [e[0] for e in evidence],
        }

    def upsert_relationship(
        self, *, a: int, b: int, category: str, fields: Record, evidence: list[int], model_call_id: int | None
    ) -> Record:
        """Writes the one relationship of a pair in a category. Null fields keep their stored value, and met_on
        keeps the earliest date, because a pair meets once. "Climbing partners since 2020" does not move the
        date that two college friends met."""
        a, b = sorted((a, b))
        with self.batch():
            row = self.conn.execute(
                "SELECT id, met_on FROM relationships "
                "WHERE run_id = ? AND person_a_id = ? AND person_b_id = ? AND category = ?",
                (self.run_id, a, b, category),
            ).fetchone()
            if row is None:
                values = {f: fields.get(f) for f in RELATIONSHIP_FIELDS}
                values["status"] = values["status"] or "current"
                values["label"] = values["label"] or category
                cursor = self.conn.execute(
                    f"INSERT INTO relationships (run_id, person_a_id, person_b_id, category, {', '.join(values)}) "
                    f"VALUES (?, ?, ?, ?, {', '.join('?' * len(values))})",
                    (self.run_id, a, b, category, *values.values()),
                )
                relationship_id = cursor.lastrowid
            else:
                relationship_id = row["id"]
                updates = {f: fields[f] for f in RELATIONSHIP_FIELDS if fields.get(f) is not None}
                if row["met_on"] and updates.get("met_on", row["met_on"]) > row["met_on"]:
                    del updates["met_on"]
                if updates:
                    self.conn.execute(
                        f"UPDATE relationships SET {', '.join(f'{k} = ?' for k in updates)}, updated_at = datetime('now') "
                        "WHERE id = ?",
                        (*updates.values(), relationship_id),
                    )
            for fact_id in evidence:
                self.conn.execute(
                    "INSERT OR IGNORE INTO relationship_evidence (relationship_id, fact_id) VALUES (?, ?)",
                    (relationship_id, fact_id),
                )
            self.show_person(a)
            self.show_person(b)
            relationship = self.relationship(relationship_id)
            if self.graph.relationships.get(relationship_id) != relationship:
                self.emit({"type": "relationship", "relationship": relationship}, model_call_id=model_call_id)
        return relationship

    def remove_relationship(self, relationship_id: int) -> None:
        with self.batch():
            self.conn.execute("DELETE FROM relationship_evidence WHERE relationship_id = ?", (relationship_id,))
            self.conn.execute("DELETE FROM relationships WHERE id = ?", (relationship_id,))
            self.emit({"type": "relationship_removed", "id": relationship_id})

    def remove_event(self, event_id: int, *, model_call_id: int | None) -> None:
        with self.batch():
            for (fact_id,) in self.conn.execute("SELECT id FROM facts WHERE event_id = ?", (event_id,)).fetchall():
                self.update_fact(fact_id, event_id=None, model_call_id=model_call_id)
            met = self.conn.execute("SELECT id FROM relationships WHERE met_event_id = ?", (event_id,)).fetchall()
            self.conn.execute("UPDATE relationships SET met_event_id = NULL WHERE met_event_id = ?", (event_id,))
            for (relationship_id,) in met:
                relationship = self.relationship(relationship_id)
                self.emit({"type": "relationship", "relationship": relationship}, model_call_id=model_call_id)
            self.conn.execute("DELETE FROM event_people WHERE event_id = ?", (event_id,))
            self.conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            self.emit({"type": "event_removed", "id": event_id}, model_call_id=model_call_id)

    def prompt(self, prompt_id: int) -> Record:
        row = self.conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        return {
            "id": row["id"],
            "text": row["text"],
            "reason": row["reason"],
            "person_ids": json.loads(row["person_ids"]),
            "shown_at_ms": row["shown_at_ms"],
            "dismissed_at_ms": row["dismissed_at_ms"],
            "payload": json.loads(row["payload_json"]),
        }

    def add_prompt(
        self,
        *,
        text: str,
        reason: str,
        person_ids: list[int],
        payload: Record | None = None,
        model_call_id: int | None = None,
    ) -> Record:
        with self.batch():
            cursor = self.conn.execute(
                "INSERT INTO prompts (run_id, text, reason, person_ids, shown_at_ms, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
                (self.run_id, text, reason, json.dumps(person_ids), self.audio_ms, json.dumps(payload or {})),
            )
            prompt = self.prompt(cursor.lastrowid)
            self.emit({"type": "prompt", "prompt": prompt}, model_call_id=model_call_id)
        return prompt

    def dismiss_prompt(self, prompt_id: int) -> None:
        with self.batch():
            self.conn.execute(
                "UPDATE prompts SET dismissed_at_ms = ? WHERE id = ? AND dismissed_at_ms IS NULL",
                (self.audio_ms, prompt_id),
            )
            self.emit({"type": "prompt", "prompt": self.prompt(prompt_id)})
