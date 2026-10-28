import asyncio
import json

from grapevine.judge import (
    DIMENSIONS,
    PAIR_SYSTEM,
    PRECISION_SYSTEM,
    RECALL_SYSTEM,
    Run,
    agree,
    findings,
    judge_pair,
    judge_run,
)
from grapevine.llm import LLM

GOLD = {
    "sample": "test",
    "people": [{"id": "p1", "name": "Ray Patel"}, {"id": "p2", "name": "Jasper Lin"}],
    "facts": [{"id": "f1", "text": "Ray works at Stripe"}],
    "traditions": [{"id": "t1", "what": "climbing on Tuesdays"}],
    "not_facts": ["Ray said okay"],
}
RUN = Run(
    listing="PEOPLE:\np1 Ray Patel",
    transcript="Ray Patel: I work at Stripe.",
    precision_items=[{"id": "r1", "text": "Ray - Jasper: friend"}, {"id": "f1", "text": "Ray works at Stripe"}],
)


def items(**verdicts: dict) -> str:
    return json.dumps({"items": [{"id": key, "note": "", **value} for key, value in verdicts.items()]})


def dimensions(**better: str) -> str:
    return json.dumps({name: {"reason": "", "better": better.get(name, "tie")} for name in DIMENSIONS})


class FakeJudgeChat:
    """Answers each judge call by its system prompt and the gold section or run order it names."""

    def __init__(self, pair: dict[str, str] | None = None) -> None:
        self.pair = pair or {}

    async def complete(self, request: dict, on_text=None) -> str:
        system, user = (message["content"] for message in request["messages"])
        if system == RECALL_SYSTEM:
            if "GOLD PEOPLE" in user:
                return items(p1={"verdict": "yes", "evidence": ["p1"]}, p2={"verdict": "partly", "evidence": []})
            if "GOLD FACTS" in user:
                return items(f1={"verdict": "no", "evidence": []})
            if "GOLD NOT_FACTS" in user:
                return items(n1={"verdict": "no", "evidence": []})
            return "not json"
        if system == PRECISION_SYSTEM:
            return items(r1={"support": "contradicted", "trivia": False}, f1={"support": "partly", "trivia": True})
        if system == PAIR_SYSTEM:
            return self.pair["first" if "RECORD A:\nfirst" in user else "second"]
        return json.dumps({name: {"reason": "", "score": 4 if name == "people" else 2} for name in DIMENSIONS})


def test_judge_run_scores_partly_as_half_and_skipped_answers_as_misses(store):
    llm = LLM(store.conn, FakeJudgeChat(), model_file_hash="test", reuse=True)
    result = asyncio.run(judge_run(llm, GOLD, RUN, run_id=store.run_id, model="m", seed=7))
    assert result["recall"] == {"people": 0.75, "facts": 0.0, "traditions": 0.0, "not_facts": 0.0}
    assert result["recall_all"] == 1.5 / 4
    assert result["precision"]["relationships"] == {"precision": 0.0, "contradicted": 1, "unsupported": 0, "trivia": 0}
    assert result["precision"]["facts"] == {"precision": 0.5, "contradicted": 0, "unsupported": 0, "trivia": 1}
    assert result["rubric_mean"] == (4 + 2 * 6) / 7


def test_agree_keeps_a_verdict_only_when_both_orders_pick_the_same_run():
    forward = json.loads(dimensions(people="A", relationships="A", timeline="B", stories="tie"))
    backward = json.loads(dimensions(people="B", relationships="A", timeline="A", stories="tie"))
    verdicts = agree(forward, backward)
    assert verdicts["people"] == "A"
    assert verdicts["relationships"] == "tie"
    assert verdicts["timeline"] == "B"
    assert verdicts["stories"] == "tie"
    assert agree(None, backward) == dict.fromkeys(DIMENSIONS, "tie")


def test_judge_pair_asks_in_both_orders(store):
    first, second = Run("first", "", []), Run("second", "", [])
    pair = {"first": dimensions(people="A", groups="A"), "second": dimensions(people="B", groups="A")}
    llm = LLM(store.conn, FakeJudgeChat(pair), model_file_hash="test", reuse=True)
    result = asyncio.run(judge_pair(llm, GOLD, first, second, run_id=store.run_id, model="m", seed=7))
    assert result["verdicts"]["people"] == "A"
    assert result["verdicts"]["groups"] == "tie"


def test_findings_list_only_misses_partial_items_and_doubtful_record_items():
    result = {
        "items": {
            "people": [
                {"id": "p1", "name": "Mia Rao", "judge": {"verdict": "yes", "note": "listed"}},
                {"id": "p2", "name": "Dana Cole", "judge": {"verdict": "no", "note": "absent"}},
            ],
            "stories": [{"id": "s1", "title": "Lisbon trip", "judge": {"verdict": "partly", "note": "no outcome"}}],
            "not_facts": [{"id": "n1", "text": "Mia said okay", "judge": {"verdict": "yes"}}],
            "precision": [
                {"id": "f1", "text": "Mia Rao lives in Seattle.", "judge": {"support": "supported", "trivia": False}},
                {"id": "f2", "text": "Mia Rao owns a boat.", "judge": {"support": "unsupported", "trivia": False}},
            ],
        }
    }

    text = findings(result)

    assert "people p2" in text and "people p1" not in text
    assert "stories s1" in text and "no outcome" in text
    assert "- Mia said okay" in text
    assert "f2 Mia Rao owns a boat.: unsupported" in text and "f1" not in text
