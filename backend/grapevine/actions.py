"""What people in the room do from the client: answer prompts and say who spoke."""

import re
from dataclasses import replace

from grapevine.extract import apply_output, held_from_payload
from grapevine.graph import Record
from grapevine.pipeline import link_speaker
from grapevine.store import Store


class ActionError(ValueError):
    pass


def active_prompt(store: Store, prompt_id: int, reason: str) -> Record:
    prompt = store.graph.prompts.get(prompt_id)
    if prompt is None or prompt["dismissed_at_ms"] is not None or prompt["reason"] != reason:
        raise ActionError(f"prompt {prompt_id} is not an open {reason} prompt")
    return prompt


def pick_person(store: Store, prompt_id: int, *, person_id: int | None, new_name: str | None) -> None:
    """Answers "Which Andrew?". The facts that waited for the answer are written now."""
    prompt = active_prompt(store, prompt_id, "which_person")
    held = held_from_payload(prompt["payload"]["held"])
    with store.batch():
        if person_id is None:
            if not new_name:
                raise ActionError("pick a person or give a new name")
            person_id = store.create_person(name=new_name, aliases=[prompt["payload"]["mention"]], gender="unknown")[
                "id"
            ]
        held.answers[prompt["payload"]["mention"]] = person_id
        for other in store.graph.active_prompts():
            same_turn = other["payload"].get("held", {}).get("teller") == prompt["payload"]["held"]["teller"]
            if other["reason"] == "which_person" and same_turn:
                store.dismiss_prompt(other["id"])
        apply_output(store, held)


def pick_speaker(
    store: Store, prompt_id: int, *, person_id: int | None, nickname: str | None, threshold: float
) -> None:
    """Answers "Which Andrew is speaking?" with a person, or with a nickname for a new person of that name."""
    prompt = active_prompt(store, prompt_id, "which_speaker")
    payload = prompt["payload"]
    held = held_from_payload(payload["held"])
    with store.batch():
        if person_id is None:
            if not nickname:
                raise ActionError("pick a person or give a nickname")
            person_id = store.create_person(name=payload["name"], aliases=[nickname], gender="unknown")["id"]
        store.dismiss_prompt(prompt_id)
        link_speaker(store, payload["label"], person_id, threshold=threshold)
        held.teller = replace(held.teller, person_id=person_id)
        apply_output(store, held)


def settle_conflict(store: Store, prompt_id: int, *, keep_fact_id: int | None) -> None:
    """Keeps one of two disputed facts, or both as disputed when `keep_fact_id` is None."""
    prompt = active_prompt(store, prompt_id, "conflict")
    fact_ids = prompt["payload"]["fact_ids"]
    with store.batch():
        if keep_fact_id is not None:
            if keep_fact_id not in fact_ids:
                raise ActionError(f"fact {keep_fact_id} is not part of this conflict")
            for fact_id in fact_ids:
                if fact_id != keep_fact_id:
                    store.update_fact(fact_id, revised_by_fact_id=keep_fact_id)
            store.update_fact(keep_fact_id, disputed=0)
        store.dismiss_prompt(prompt_id)


def rename_person(store: Store, person_id: int, name: str) -> None:
    """Renames a person of the run, such as a name that the transcript misheard. The old name stays as an alias,
    so the same mishearing still finds the person, and the facts about the person use the new name."""
    person = store.graph.people.get(person_id)
    if person is None:
        raise ActionError(f"person {person_id} is not in this run")
    old = person["name"]
    if name == old:
        return
    if any(p["name"].lower() == name.lower() for p in store.graph.people.values() if p["id"] != person_id):
        raise ActionError(f"{name} is already in this run")
    if store.name_taken(name, person_id):
        raise ActionError(f"Someone from another session is already called {name}")
    with store.batch():
        store.update_person(person_id, add_aliases=[old, name.split()[0]], gender=None, name=name)
        for fact in list(store.graph.facts.values()):
            about = fact["told_by_person_id"] == person_id or any(p["person_id"] == person_id for p in fact["people"])
            text = renamed(fact["text"], old, name)
            if about and text != fact["text"]:
                store.update_fact(fact["id"], text=text)
        for rel in list(store.graph.relationships.values()):
            summary = renamed(rel["summary"] or "", old, name)
            if person_id in (rel["a"], rel["b"]) and summary != (rel["summary"] or ""):
                fields = {"summary": summary}
                store.upsert_relationship(
                    a=rel["a"], b=rel["b"], category=rel["category"], fields=fields, evidence=[], model_call_id=None
                )


def renamed(text: str, old: str, new: str) -> str:
    """Writes the new name for the old full name, then the new first name for the old first name."""
    text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    old_first, new_first = old.split()[0], new.split()[0]
    if old_first == new_first:
        return text
    return re.sub(rf"\b{re.escape(old_first)}\b", new_first, text)


def assign_speaker(
    store: Store, segment_id: int, *, person_id: int | None, new_name: str | None, every_line: bool, threshold: float
) -> list[int]:
    """Says by hand who spoke a turn, and returns the turns whose speaker changed. With `every_line`, the voice
    of the turn is that person, which fixes an introduction that named the wrong voice. Turns set by hand keep
    their speaker when the pipeline matches voices again. A new name adds a person of that name."""
    segment = store.graph.segments.get(segment_id)
    if segment is None:
        raise ActionError(f"segment {segment_id} is not in this run")
    if (person_id is None) == (new_name is None):
        raise ActionError("pick a person or give a new name, not both")
    if (
        person_id is not None
        and store.conn.execute("SELECT 1 FROM people WHERE id = ?", (person_id,)).fetchone() is None
    ):
        raise ActionError(f"no person {person_id}")
    label = segment["speaker_label"]
    with store.batch():
        if person_id is None:
            person_id = store.create_person(name=new_name, aliases=[], gender="unknown")["id"]
        store.show_person(person_id)
        if every_line:
            old = store.graph.speakers.get(label)
            store.set_speaker(label, person_id, None)
            lines = [
                s
                for s in store.graph.segments.values()
                if s["speaker_label"] == label and s["person_id"] in (old, None)
            ]
        else:
            lines = [segment]
            if store.graph.speakers.get(label) is None:
                link_speaker(store, label, person_id, threshold=threshold)
        changed = [s["id"] for s in lines if s["person_id"] != person_id]
        for line in lines:
            store.update_segment(line["id"], person_id=person_id, speaker_manual=True)
        for fact in list(store.graph.facts.values()):
            if set(changed) & set(fact["segment_ids"]) and fact["told_by_person_id"] is None:
                store.update_fact(fact["id"], told_by_person_id=person_id)
    return changed
