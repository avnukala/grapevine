import { useEffect, useMemo, useRef, useState } from "react";
import { api, audioUrl } from "../api";
import { personName } from "../state/derive";
import type { WeightFactors } from "../state/derive";
import { recent, speakerAt } from "../state/pulses";
import { stateAt } from "../state/reducer";
import type { Change, ModelCall, Session } from "../state/types";
import { useRun } from "../state/useRun";
import { GraphView } from "./GraphView";
import type { View } from "./LiveScreen";
import { TimelineView } from "./TimelineView";
import { Transcript } from "./Transcript";

const SPEEDS = [0.5, 1, 1.5, 2, 4];
type Tab = "transcript" | "calls" | "changes";

interface DebugScreenProps {
  factors: WeightFactors;
}

// Dev and test only. Replays a run against its audio: the graph and timeline show the state at the
// current audio time, next to the transcript, the model calls, and the change log.
export function DebugScreen({ factors }: DebugScreenProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [runId, setRunId] = useState<number | null>(() => Number(new URLSearchParams(location.hash.split("?")[1]).get("run")) || null);
  const [calls, setCalls] = useState<ModelCall[]>([]);
  const [ms, setMs] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [view, setView] = useState<View>("graph");
  const [tab, setTab] = useState<Tab>("transcript");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const audio = useRef<HTMLAudioElement>(null);
  const live = useRun(runId);

  useEffect(() => {
    api.sessions().then(setSessions).catch(() => setSessions([]));
  }, [live.status]);
  useEffect(() => {
    setCalls([]);
    if (runId !== null) api.modelCalls(runId).then(setCalls).catch(() => setCalls([]));
  }, [runId, live.status]);
  useEffect(() => {
    if (audio.current) audio.current.playbackRate = speed;
  }, [speed]);

  const session = sessions.find((s) => s.runs.some((r) => r.id === runId)) ?? null;
  const run = session?.runs.find((r) => r.id === runId);
  const duration = Math.max(
    (audio.current?.duration ?? 0) * 1000 || 0,
    ...live.changes.map((c) => c.audio_ms),
    1000,
  );
  const state = useMemo(() => stateAt(live.changes, ms), [live.changes, ms]);
  const changed = useMemo(() => recent(live.changes, (c) => c.audio_ms, ms), [live.changes, ms]);
  const timings = useMemo(() => latencySummary(Object.values(live.state.segments).map((s) => s.timings)), [live.state]);

  const seek = (to: number) => {
    setMs(to);
    if (audio.current) audio.current.currentTime = to / 1000;
  };

  return (
    <div className="screen debug" data-testid="debug">
      <div className="controls">
        <select value={runId ?? ""} onChange={(e) => setRunId(Number(e.target.value) || null)} aria-label="Run">
          <option value="">Pick a run</option>
          {sessions.flatMap((s) =>
            s.runs.map((r) => (
              <option key={r.id} value={r.id}>
                {s.name}: run {r.id} ({r.kind}, {r.status}
                {r.accepted ? ", accepted" : ""})
              </option>
            )),
          )}
        </select>
        {session?.audio_uri && (
          <audio
            ref={audio}
            src={audioUrl(session.id)}
            controls
            onTimeUpdate={(e) => setMs(Math.round(e.currentTarget.currentTime * 1000))}
            onLoadedMetadata={(e) => (e.currentTarget.playbackRate = speed)}
          />
        )}
        <label>
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <input
          className="scrubber"
          type="range"
          min={0}
          max={Math.ceil(duration)}
          value={Math.min(ms, duration)}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Audio time"
          data-testid="scrubber"
        />
        <span className="clock" data-testid="clock">
          {(ms / 1000).toFixed(1)}s / {(duration / 1000).toFixed(1)}s
        </span>
      </div>

      {run && (
        <div className="run-meta">
          Run {run.id} · {run.kind} · {live.status ?? run.status} · code {run.code_version} · detect {timings.detect_ms} ·
          transcribe {timings.transcribe_ms} · extract {timings.extract_ms} · speech end to facts {timings.total_ms}
        </div>
      )}

      <div className="debug-body">
        <div className="stage">
          <div className="view-switch">
            <button className={view === "graph" ? "on" : ""} onClick={() => setView("graph")}>
              Graph
            </button>
            <button className={view === "timeline" ? "on" : ""} onClick={() => setView("timeline")}>
              Timeline
            </button>
          </div>
          {view === "graph" ? (
            <GraphView
              state={state}
              pulses={changed.pulses}
              speakingId={speakerAt(state, ms)}
              factors={factors}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          ) : (
            <TimelineView state={state} recentFactIds={changed.factIds} recentEventIds={changed.eventIds} />
          )}
        </div>

        <div className="side">
          <div className="tabs">
            {(["transcript", "calls", "changes"] as Tab[]).map((t) => (
              <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
                {t === "calls" ? `Model calls (${calls.length})` : t === "changes" ? `Changes (${live.changes.length})` : "Transcript"}
              </button>
            ))}
          </div>
          {tab === "transcript" && (
            <Transcript
              state={live.state}
              voiceThreshold={Number(run?.config?.voice_threshold ?? 0.75)}
              currentMs={ms}
              showTimings
              onSeek={seek}
            />
          )}
          {tab === "calls" && <ModelCalls calls={calls} onSeek={seek} />}
          {tab === "changes" && <ChangeLog changes={live.changes} currentMs={ms} onSeek={seek} names={live.state} />}
        </div>
      </div>
    </div>
  );
}

function latencySummary(timings: Record<string, number>[]): Record<string, string> {
  const steps = ["detect_ms", "transcribe_ms", "extract_ms", "total_ms"];
  return Object.fromEntries(
    steps.map((step) => {
      const values = timings.map((t) => t[step]).filter((v): v is number => v !== undefined).sort((x, y) => x - y);
      return [step, values.length ? `median ${values[Math.floor(values.length / 2)]}ms` : "n/a"];
    }),
  );
}

interface ModelCallsProps {
  calls: ModelCall[];
  onSeek: (ms: number) => void;
}

function ModelCalls({ calls, onSeek }: ModelCallsProps) {
  return (
    <ol className="calls" data-testid="model-calls">
      {calls.map((call) => (
        <li key={call.id}>
          <details>
            <summary>
              #{call.id} {call.worker} · {call.cached ? "reused" : `${call.latency_ms}ms`} · {call.prompt_version}
              {call.audio_ms !== null && (
                <button className="link" onClick={() => onSeek(call.audio_ms!)}>
                  @{(call.audio_ms / 1000).toFixed(1)}s
                </button>
              )}
            </summary>
            <div className="muted">
              model {call.model} · file {call.model_file_hash} · seed {call.seed} · input {call.input_hash.slice(0, 12)}
            </div>
            <pre>{call.request.messages[1]?.content}</pre>
            <pre>{prettyJson(call.response)}</pre>
          </details>
        </li>
      ))}
    </ol>
  );
}

function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

interface ChangeLogProps {
  changes: Change[];
  currentMs: number;
  names: Parameters<typeof personName>[0];
  onSeek: (ms: number) => void;
}

function ChangeLog({ changes, currentMs, names, onSeek }: ChangeLogProps) {
  return (
    <ol className="changes" data-testid="change-log">
      {changes.map((change) => (
        <li key={change.seq} className={change.audio_ms > currentMs ? "future" : ""}>
          <button className="link" onClick={() => onSeek(change.audio_ms)}>
            {(change.audio_ms / 1000).toFixed(1)}s
          </button>{" "}
          #{change.seq} {describe(change, names)}
        </li>
      ))}
    </ol>
  );
}

function describe(change: Change, state: Parameters<typeof personName>[0]): string {
  const op = change.op;
  switch (op.type) {
    case "person":
      return `person ${op.person.name}${op.person.merged_into_id ? " merged" : ""}`;
    case "speaker":
      return `voice ${op.speaker.label} is ${personName(state, op.speaker.person_id)}`;
    case "segment":
      return `segment ${op.segment.speaker_label}: ${op.segment.text.slice(0, 60)}`;
    case "fact":
      return `fact [${op.fact.kind}] ${op.fact.text}${op.fact.disputed ? " (disputed)" : ""}${op.fact.revised_by_fact_id ? " (revised)" : ""}`;
    case "event":
      return `event ${op.event.name} ${op.event.start_date ?? ""}`;
    case "group":
      return `group ${op.group.name}`;
    case "member":
      return `member ${personName(state, op.member.person_id)} in group ${op.member.group_id}`;
    case "relationship":
      return `relationship ${personName(state, op.relationship.a)} - ${personName(state, op.relationship.b)}: ${op.relationship.category} "${op.relationship.label}" ${op.relationship.status}`;
    case "relationship_removed":
      return `relationship ${op.id} removed`;
    case "prompt":
      return `prompt [${op.prompt.reason}] ${op.prompt.text}${op.prompt.dismissed_at_ms !== null ? " (dismissed)" : ""}`;
    case "phase":
      return `phase ${op.phase}`;
  }
}
