"""Command line: replay a recording, score a run, or serve the API."""

import argparse
import asyncio
import json
import logging
import os
import statistics
from pathlib import Path

from grapevine import db
from grapevine.audio import file_chunks
from grapevine.compare import differences, judge
from grapevine.config import DB_PATH, ROOT, run_config
from grapevine.llm import LLM
from grapevine.merge import merge
from grapevine.runs import (
    chat,
    create_run,
    llm_file_hash,
    load_models,
    open_store,
    run_pipeline,
    session_for_audio,
    set_status,
    start_pipeline,
)
from grapevine.score import score
from grapevine.store import Store


def parse_overrides(pairs: list[str]) -> dict:
    overrides: dict = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        target = overrides
        if key.startswith("turn_rules."):
            target = overrides.setdefault("turn_rules", {})
            key = key.removeprefix("turn_rules.")
        target[key] = json.loads(value) if value[:1] in "0123456789-[{tfn" else value
    return overrides


async def replay(audio: Path, *, speed: float | None, overrides: dict) -> int:
    conn = db.connect(DB_PATH)
    config = run_config(overrides)
    session = session_for_audio(conn, audio)
    run_id = create_run(conn, session_id=session["id"], kind="replay", config=config)
    print(f"run {run_id} on session {session['id']} ({session['name']})")
    active = start_pipeline(
        conn, run_id=run_id, session=session, config=config, models=load_models(config), broadcast=None
    )
    await run_pipeline(conn, active, file_chunks(audio, speed=speed, stop=active.stop))
    return run_id


def accept(run_id: int) -> None:
    """Marks a run as the accepted run of its session, so the next merge reads it."""
    conn = db.connect(DB_PATH)
    session_id = conn.execute("SELECT session_id FROM runs WHERE id = ?", (run_id,)).fetchone()[0]
    conn.execute("BEGIN")
    conn.execute("UPDATE runs SET accepted = 0 WHERE session_id = ?", (session_id,))
    conn.execute("UPDATE runs SET accepted = 1 WHERE id = ?", (run_id,))
    conn.execute("COMMIT")
    print(f"accepted run {run_id} for session {session_id}")


async def merge_accepted() -> None:
    conn = db.connect(DB_PATH)
    config = run_config({})
    store = Store(conn, create_run(conn, session_id=None, kind="merge", config=config))
    llm = LLM(conn, chat(), model_file_hash=llm_file_hash(), reuse=config.reuse_model_calls)
    await merge(store, llm, model=config.summary_model, seed=config.seed)
    set_status(conn, store.run_id, "done")
    graph = store.graph
    print(
        f"merge run {store.run_id}: {len(graph.people)} people, {len(graph.relationships)} relationships, "
        f"{len(graph.facts)} facts, {len(graph.active_prompts())} to review"
    )
    for prompt in graph.active_prompts():
        print(f"  [{prompt['reason']}] {prompt['text']}")


async def compare_runs(first: int, second: int, *, with_judge: bool) -> None:
    conn = db.connect(DB_PATH)
    a, b = open_store(conn, first).graph, open_store(conn, second).graph
    for key, diff in differences(a, b).items():
        print(f"{key}: {len(diff['only_a'])} only in run {first}, {len(diff['only_b'])} only in run {second}")
        for item in diff["only_a"]:
            print(f"  - {item}")
        for item in diff["only_b"]:
            print(f"  + {item}")
    if with_judge:
        config = run_config({})
        llm = LLM(conn, chat(), model_file_hash=llm_file_hash(), reuse=config.reuse_model_calls)
        verdict = await judge(llm, a, b, run_id=second, model=config.summary_model, seed=config.seed)
        print(f"judge (A = run {first}, B = run {second}):", json.dumps(verdict, indent=2))


def report(run_id: int) -> None:
    conn = db.connect(DB_PATH)
    store = open_store(conn, run_id)
    graph = store.graph
    run = conn.execute(
        "SELECT runs.*, sessions.audio_uri FROM runs JOIN sessions ON sessions.id = runs.session_id WHERE runs.id = ?",
        (run_id,),
    ).fetchone()
    names = {pid: p["name"] for pid, p in graph.people.items()}
    print(
        f"\nrun {run_id}: {len(graph.segments)} segments, {len(graph.people)} people, {len(graph.facts)} facts, "
        f"{len(graph.relationships)} relationships, {len(graph.events)} events, {len(graph.prompts)} prompts"
    )
    for segment in sorted(graph.segments.values(), key=lambda s: s["start_ms"]):
        who = names.get(segment["person_id"], f"unknown {segment['speaker_label']}")
        print(
            f"  {segment['start_ms'] / 1000:6.1f}s {who:15} {segment['speaker_confidence'] or 0:.2f}  {segment['text']}"
        )
    totals = [s["timings"]["total_ms"] for s in graph.segments.values() if "total_ms" in s["timings"]]
    extracts = [s["timings"]["extract_ms"] for s in graph.segments.values() if "extract_ms" in s["timings"]]
    if totals:
        print(
            f"  speech end to facts: median {statistics.median(totals)} ms, max {max(totals)} ms; "
            f"extraction median {statistics.median(extracts)} ms"
        )
    print("facts:")
    for fact in graph.facts.values():
        flags = (" REVISED" if fact["revised_by_fact_id"] else "") + (" DISPUTED" if fact["disputed"] else "")
        print(f"  f{fact['id']} [{fact['kind']}, {fact['start_date'] or '-'}] {fact['text']}{flags}")
    print("relationships:")
    for rel in graph.relationships.values():
        print(
            f"  {names[rel['a']]} - {names[rel['b']]}: {rel['category']} '{rel['label']}' {rel['status']}"
            f" met {rel['met_on']} c{rel['closeness']} r{rel['recency']} s{rel['sentiment']}"
        )
    print("events:", [f"{e['name']} ({e['start_date']})" for e in graph.events.values()])
    print(
        "groups:",
        [
            (g["name"], [names.get(m["person_id"]) for m in graph.members.values() if m["group_id"] == g["id"]])
            for g in graph.groups.values()
        ],
    )
    print("prompts:", [f"{p['reason']}: {p['text']}" for p in graph.prompts.values()])
    audio = Path(run["audio_uri"])
    audio = audio if audio.is_absolute() else ROOT / audio
    if audio.with_name(audio.name.rsplit(".", 1)[0] + ".labels.json").exists():
        print("score:", json.dumps(score(graph, audio), indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="grapevine")
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("replay", help="run the pipeline over an audio file")
    replay_parser.add_argument("audio", type=Path)
    replay_parser.add_argument("--speed", type=float, help="stream in real time at this speed; default is offline")
    replay_parser.add_argument("--set", action="append", default=[], help="override a run setting, key=value")
    report_parser = commands.add_parser("report", help="print and score a run")
    report_parser.add_argument("run_id", type=int)
    accept_parser = commands.add_parser("accept", help="accept a run for its session")
    accept_parser.add_argument("run_id", type=int)
    commands.add_parser("merge", help="merge the accepted runs into the global graph")
    compare_parser = commands.add_parser("compare", help="compare two runs on the same session")
    compare_parser.add_argument("first", type=int)
    compare_parser.add_argument("second", type=int)
    compare_parser.add_argument("--judge", action="store_true", help="also ask the local model which run is better")
    serve_parser = commands.add_parser("serve", help="serve the API and WebSocket")
    serve_parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8790")))
    args = parser.parse_args()

    match args.command:
        case "replay":
            run_id = asyncio.run(replay(args.audio, speed=args.speed, overrides=parse_overrides(args.set)))
            report(run_id)
        case "report":
            report(args.run_id)
        case "accept":
            accept(args.run_id)
        case "merge":
            asyncio.run(merge_accepted())
        case "compare":
            asyncio.run(compare_runs(args.first, args.second, with_judge=args.judge))
        case "serve":
            import uvicorn

            uvicorn.run("grapevine.server:app", port=args.port, log_level="info", timeout_graceful_shutdown=3)
