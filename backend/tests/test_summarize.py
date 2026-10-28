import asyncio

from grapevine.llm import Call, Result
from grapevine.summarize import summarize

STORY = {
    "event": None,
    "title": "The Seattle Blackout",
    "lines": ["The lights died.", "Dana Cole: We sat in the dark all night."],
    "text": "The power went out in Seattle.",
    "matters": "They became friends that night.",
    "tellers": ["p1", "p2"],
    "people": ["p1", "p2"],
}


class Summarizer:
    def __init__(self, stories: list[dict]) -> None:
        self.calls: list[Call] = []
        self.stories = stories

    async def call(self, call: Call, **_) -> Result:
        self.calls.append(call)
        return Result({"stories": self.stories, "relationships": []}, None)


def told(store):
    mia = store.create_person(name="Mia Rao", aliases=[], gender="female")
    dana = store.create_person(name="Dana Cole", aliases=[], gender="female")
    segment = store.add_segment(
        start_ms=0, end_ms=1, speaker_label="A", person_id=mia["id"], confidence=1, text="The lights died.", timings={}
    )
    store.add_segment(
        start_ms=1,
        end_ms=2,
        speaker_label="B",
        person_id=dana["id"],
        confidence=1,
        text="We sat in the dark all night.",
        timings={},
    )
    store.add_fact({"kind": "story", "text": "Power cut"}, people=[], segment_ids=[segment["id"]], model_call_id=None)
    return mia, dana


def test_a_story_keeps_its_tellers_why_it_matters_and_reads_the_transcript(store):
    mia, dana = told(store)
    llm = Summarizer([STORY])

    asyncio.run(summarize(store, llm, model="m", seed=1))

    assert "Mia Rao: The lights died." in llm.calls[0].user
    story = next(f for f in store.graph.facts.values() if f["text"] == "The Seattle Blackout")
    assert story["told_by_person_id"] == mia["id"]
    assert story["detail"] == (
        "The power went out in Seattle.\n\nThey became friends that night.\n\nAlso told by Dana Cole."
    )
    assert dana["id"] in store.graph.people


def test_a_story_drops_when_its_lines_are_not_in_the_transcript(store):
    told(store)
    made_up = {**STORY, "title": "The Lisbon Race", "lines": ["The lights died.", "Mia won the race twice."]}
    stub = {**STORY, "title": "Mia Rao Is In The League", "lines": ["The lights died."]}
    one_line = {**STORY, "lines": ["The lights died. We sat in the dark all night."]}

    asyncio.run(summarize(store, Summarizer([made_up, stub, one_line]), model="m", seed=1))

    titles = {f["text"] for f in store.graph.facts.values() if f["kind"] == "story" and not f["segment_ids"]}
    assert titles == {"The Seattle Blackout"}


def test_the_story_count_is_capped_so_a_loop_still_ends_in_valid_json(store):
    told(store)
    llm = Summarizer([STORY])

    asyncio.run(summarize(store, llm, model="m", seed=1))

    assert llm.calls[0].schema["properties"]["stories"]["maxItems"] == 12
