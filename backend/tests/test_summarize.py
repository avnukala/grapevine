import asyncio

from grapevine.llm import Call, Result
from grapevine.summarize import summarize

STORY = {
    "event": None,
    "title": "The Seattle Blackout",
    "text": "The power went out in Seattle.",
    "matters": "They became friends that night.",
    "tellers": ["p1", "p2"],
    "people": ["p1", "p2"],
}


class Summarizer:
    def __init__(self) -> None:
        self.calls: list[Call] = []

    async def call(self, call: Call, **_) -> Result:
        self.calls.append(call)
        return Result({"stories": [STORY], "relationships": []}, None)


def test_a_story_keeps_its_tellers_why_it_matters_and_reads_the_transcript(store):
    mia = store.create_person(name="Mia Rao", aliases=[], gender="female")
    dana = store.create_person(name="Dana Cole", aliases=[], gender="female")
    segment = store.add_segment(
        start_ms=0, end_ms=1, speaker_label="A", person_id=mia["id"], confidence=1, text="The lights died.", timings={}
    )
    store.add_fact({"kind": "story", "text": "Power cut"}, people=[], segment_ids=[segment["id"]], model_call_id=None)
    llm = Summarizer()

    asyncio.run(summarize(store, llm, model="m", seed=1))

    assert "Mia Rao: The lights died." in llm.calls[0].user
    story = next(f for f in store.graph.facts.values() if f["text"] == "The Seattle Blackout")
    assert story["told_by_person_id"] == mia["id"]
    assert story["detail"] == (
        "The power went out in Seattle.\n\nThey became friends that night.\n\nTold by Mia Rao and Dana Cole."
    )
    assert dana["id"] in store.graph.people
