"""What people in the room do from the client: answer prompts and say who spoke."""

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


def assign_speaker(store: Store, segment_id: int, *, person_id: int, threshold: float) -> None:
    """Says who spoke an unknown turn. The voice keeps that name for the rest of the run."""
    segment = store.graph.segments.get(segment_id)
    if segment is None:
        raise ActionError(f"segment {segment_id} is not in this run")
    with store.batch():
        store.show_person(person_id)
        store.update_segment(segment_id, person_id=person_id)
        if store.graph.speakers.get(segment["speaker_label"]) is None:
            link_speaker(store, segment["speaker_label"], person_id, threshold=threshold)
        for fact in list(store.graph.facts.values()):
            if segment_id in fact["segment_ids"] and fact["told_by_person_id"] is None:
                store.update_fact(fact["id"], told_by_person_id=person_id)
