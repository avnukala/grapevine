"""Scores a run against the hand labels and expected facts of a sample clip."""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from grapevine.graph import Graph, Record

FRAME_MS = 10


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref_word != hyp_word)))
        previous = current
    return previous[-1] / max(1, len(reference))


def similar(expected: str, found: str) -> bool:
    """Group names match loosely, because Whisper hears "Sunday Sends" as "Sunday Sands"."""
    a, b = " ".join(words(expected)).removeprefix("the "), " ".join(words(found)).removeprefix("the ")
    return SequenceMatcher(None, a, b).ratio() >= 0.75 or a in b or b in a


def same_person(name: str, label: str) -> bool:
    """Transcripts misspell surnames, so a matching first name counts."""
    return name == label or name.split()[:1] == label.split()[:1]


def speakers(graph: Graph, labels: Record) -> Record:
    """Share of labeled speech time where the run names the right person, names nobody, or names someone else."""
    names = {pid: p["name"] for pid, p in graph.people.items()}
    right = unknown = wrong = missed = 0
    segments = sorted(graph.segments.values(), key=lambda s: s["start_ms"])
    for turn in labels["turns"]:
        for ms in range(turn["start_ms"], turn["end_ms"], FRAME_MS):
            segment = next((s for s in segments if s["start_ms"] <= ms < s["end_ms"]), None)
            if segment is None:
                missed += 1
            elif segment["person_id"] is None:
                unknown += 1
            elif same_person(names.get(segment["person_id"], ""), turn["speaker"]):
                right += 1
            else:
                wrong += 1
    total = max(1, right + unknown + wrong + missed)
    return {
        "right": round(right / total, 3),
        "unknown": round(unknown / total, 3),
        "wrong": round(wrong / total, 3),
        "missed": round(missed / total, 3),
    }


def facts(graph: Graph, expected: Record) -> Record:
    by_name = {p["name"].lower(): pid for pid, p in graph.people.items()}

    def person(name: str) -> int | None:
        if name.lower() in by_name:
            return by_name[name.lower()]
        first = [pid for n, pid in by_name.items() if n.split()[0] == name.lower().split()[0]]
        return first[0] if len(first) == 1 else None

    found_people = [n for n in expected["people"] if person(n) is not None]
    found_relationships = []
    for rel in expected["relationships"]:
        a, b = person(rel["a"]), person(rel["b"])
        match = next(
            (
                r
                for r in graph.relationships.values()
                if {r["a"], r["b"]} == {a, b} and r["category"] == rel["category"]
            ),
            None,
        )
        if (
            match
            and match["status"] == rel["status"]
            and ("met" not in rel or (match["met_on"] or "").startswith(rel["met"]))
        ):
            found_relationships.append(f"{rel['a']} - {rel['b']} {rel['category']}")
    event_years = [int(e["start_date"][:4]) for e in graph.events.values() if e["start_date"]]
    found_events = [e["name"] for e in expected["events"] if e["year"] in event_years]
    found_groups = []
    for group in expected["groups"]:
        ids = [g["id"] for g in graph.groups.values() if similar(group["name"], g["name"])]
        members = {m["person_id"] for m in graph.members.values() if m["group_id"] in ids}
        if {person(n) for n in group["members"]} <= members:
            found_groups.append(group["name"])
    disputes = sum(p["reason"] == "conflict" for p in graph.prompts.values())
    ambiguous = [p["payload"].get("mention") for p in graph.prompts.values() if p["reason"] == "which_person"]
    return {
        "people": f"{len(found_people)}/{len(expected['people'])}",
        "relationships": f"{len(found_relationships)}/{len(expected['relationships'])}",
        "relationships_found": found_relationships,
        "events": f"{len(found_events)}/{len(expected['events'])}",
        "groups": f"{len(found_groups)}/{len(expected['groups'])}",
        "disputes": f"{min(disputes, expected['disputes'])}/{expected['disputes']}",
        "ambiguous_names": f"{len(set(ambiguous) & set(expected['ambiguous_names']))}/{len(expected['ambiguous_names'])}",
    }


def score(graph: Graph, audio: Path) -> Record:
    stem = audio.name.rsplit(".", 1)[0]
    labels = json.loads(audio.with_name(f"{stem}.labels.json").read_text())
    script = json.loads(audio.with_name(f"{stem}.script.json").read_text())
    reference = [w for t in labels["turns"] for w in words(t["text"])]
    hypothesis = [w for s in sorted(graph.segments.values(), key=lambda s: s["start_ms"]) for w in words(s["text"])]
    return {
        "speakers": speakers(graph, labels),
        "word_error_rate": round(word_error_rate(reference, hypothesis), 3),
        "facts": facts(graph, script["expected"]),
    }
