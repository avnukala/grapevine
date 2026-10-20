import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { CATEGORIES, CATEGORY_LIST } from "../state/categories";
import { personName, roomIds } from "../state/derive";
import type { WeightFactors } from "../state/derive";
import { recent } from "../state/pulses";
import type { Prompt, Segment, Session } from "../state/types";
import { useNow, useRun } from "../state/useRun";
import type { TimedChange } from "../state/useRun";
import { GraphView } from "./GraphView";
import { PersonDetails } from "./PersonDetails";
import type { Answer } from "./PromptCard";
import { PromptPanel } from "./PromptPanel";
import { SessionControls } from "./SessionControls";
import { TimelineView } from "./TimelineView";
import { Transcript } from "./Transcript";
import { WeightPanel } from "./WeightPanel";

export type View = "graph" | "timeline";
const GLOW_MS = 2500;

interface LiveScreenProps {
  factors: WeightFactors;
  onFactors: (factors: WeightFactors) => void;
}

export function LiveScreen({ factors, onFactors }: LiveScreenProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [runId, setRunId] = useState<number | null>(null);
  const [view, setView] = useState<View>("graph");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const live = useRun(runId);
  const now = useNow();
  const { state } = live;

  const refresh = useCallback(async () => {
    setSessions(await api.sessions());
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh, live.status]);

  const session = sessions.find((s) => s.id === sessionId) ?? null;
  const run = session?.runs.find((r) => r.id === runId);
  const runActive = live.status === "running" || (run?.status === "running" && live.status === null);
  const threshold = Number(run?.config?.voice_threshold ?? 0.75);

  const changed = useMemo(
    () => recent(live.changes, (c) => (c as TimedChange).receivedAt, now),
    [live.changes, now],
  );
  const lastSegment = Object.values(state.segments).sort((x, y) => y.id - x.id)[0];
  const lastSegmentAt = live.changes.find((c) => c.op.type === "segment" && c.op.segment.id === lastSegment?.id)?.receivedAt;
  const speakingId =
    live.speaking.personId ??
    (lastSegment && lastSegmentAt && now - lastSegmentAt < GLOW_MS ? lastSegment.person_id : null);

  async function guard(task: () => Promise<unknown>) {
    setError(null);
    try {
      await task();
    } catch (e) {
      setError(String(e));
    }
  }
  const onAnswer = (prompt: Prompt, answer: Answer) => guard(() => api.answer(runId!, prompt.id, answer as never));
  const onDismiss = (prompt: Prompt) => guard(() => api.dismiss(runId!, prompt.id));
  const onAssign = (segment: Segment, answer: { person_id: number } | { new_name: string }) =>
    guard(() => api.assignSpeaker(runId!, segment.id, answer));

  return (
    <div className="screen live">
      <SessionControls
        sessions={sessions}
        sessionId={sessionId}
        runId={runId}
        runActive={runActive}
        phase={state.phase}
        onSession={(id) => {
          setSessionId(id);
          setRunId(sessions.find((s) => s.id === id)?.runs[0]?.id ?? null);
        }}
        onRun={setRunId}
        onRefresh={refresh}
        onError={setError}
      />
      {(error || live.error) && <div className="error">{error ?? live.error}</div>}

      <div className="stage">
        <div className="view-switch">
          <button className={view === "graph" ? "on" : ""} onClick={() => setView("graph")}>
            Graph
          </button>
          <button className={view === "timeline" ? "on" : ""} onClick={() => setView("timeline")}>
            Timeline
          </button>
          <span className="status" data-testid="run-status">
            {runId ? `Run ${runId} · ${live.status ?? run?.status ?? ""} · ${state.phase}` : "No run"}
          </span>
        </div>

        {view === "graph" ? (
          <GraphView
            state={state}
            pulses={changed.pulses}
            speakingId={speakingId}
            factors={factors}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        ) : (
          <TimelineView state={state} recentFactIds={changed.factIds} recentEventIds={changed.eventIds} />
        )}

        {state.phase === "intro" && runId !== null && (
          <div className="intro-card" data-testid="intro">
            <strong>Introductions.</strong> Each person says their name and a few sentences.
            <ul>
              {roomIds(state).map((id) => (
                <li key={id}>{personName(state, id)}</li>
              ))}
            </ul>
          </div>
        )}

        <PromptPanel state={state} onAnswer={onAnswer} onDismiss={onDismiss} />
        {selectedId !== null && (
          <PersonDetails state={state} personId={selectedId} onClose={() => setSelectedId(null)} />
        )}

        <div className="bottom">
          <Transcript state={state} voiceThreshold={threshold} limit={3} onAssign={onAssign} />
          {runActive && <TypedTurn runId={runId!} onError={setError} />}
          <div className="legend">
            {CATEGORY_LIST.map((c) => (
              <span key={c}>
                <span className="swatch" style={{ background: CATEGORIES[c].color }} />
                {c}
              </span>
            ))}
            <details>
              <summary>Edge weight</summary>
              <WeightPanel factors={factors} onChange={onFactors} />
            </details>
          </div>
        </div>
      </div>
    </div>
  );
}

interface TypedTurnProps {
  runId: number;
  onError: (message: string | null) => void;
}

function TypedTurn({ runId, onError }: TypedTurnProps) {
  const [speaker, setSpeaker] = useState("");
  const [text, setText] = useState("");
  return (
    <form
      className="typed-turn"
      onSubmit={(e) => {
        e.preventDefault();
        if (!speaker.trim() || !text.trim()) return;
        api
          .text(runId, speaker, text)
          .then(() => setText(""))
          .catch((err) => onError(String(err)));
      }}
    >
      <input value={speaker} onChange={(e) => setSpeaker(e.target.value)} placeholder="Speaker" aria-label="Speaker" />
      <input value={text} onChange={(e) => setText(e.target.value)} placeholder="What they said" aria-label="Turn" />
      <button>Add turn</button>
    </form>
  );
}
