import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { WeightFactors } from "../state/derive";
import type { Prompt, Run, Session } from "../state/types";
import { useRun } from "../state/useRun";
import { GraphView } from "./GraphView";
import type { View } from "./LiveScreen";
import { PersonDetails } from "./PersonDetails";
import { PromptCard } from "./PromptCard";
import type { Answer } from "./PromptCard";
import { TimelineView } from "./TimelineView";

interface GlobalScreenProps {
  factors: WeightFactors;
}

// The graph across sessions: facts of accepted runs and the relationships of the latest merge run.
export function GlobalScreen({ factors }: GlobalScreenProps) {
  const [mergeRun, setMergeRun] = useState<Run | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [view, setView] = useState<View>("graph");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { state } = useRun(mergeRun?.id ?? null);

  const load = useCallback(async () => {
    const [global, all] = await Promise.all([api.global(), api.sessions()]);
    setMergeRun(global.run);
    setSessions(all);
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  const accepted = sessions.flatMap((s) => s.runs.filter((r) => r.accepted).map((r) => ({ session: s, run: r })));
  const review = Object.values(state.prompts).filter((p) => p.dismissed_at_ms === null);

  async function runMerge() {
    setBusy(true);
    setError(null);
    try {
      setMergeRun(await api.merge());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const onAnswer = (prompt: Prompt, answer: Answer) =>
    api.resolveMerge(mergeRun!.id, prompt.id, answer as { action: string; fact_id?: number }).catch((e) => setError(String(e)));

  return (
    <div className="screen global" data-testid="global">
      <div className="controls">
        <button disabled={busy || accepted.length === 0} onClick={() => void runMerge()}>
          {busy ? "Merging…" : "Merge accepted runs"}
        </button>
        <span className="muted">
          Accepted: {accepted.map(({ session, run }) => `${session.name} (run ${run.id})`).join(", ") || "none yet"}
        </span>
        {mergeRun && <span className="muted">Showing merge run {mergeRun.id}</span>}
      </div>
      {error && <div className="error">{error}</div>}
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
              pulses={[]}
              speakingId={null}
              factors={factors}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          ) : (
            <TimelineView state={state} recentFactIds={[]} recentEventIds={[]} />
          )}
          {selectedId !== null && (
            <PersonDetails state={state} personId={selectedId} onClose={() => setSelectedId(null)} />
          )}
        </div>
        <div className="side">
          <h3>Review ({review.length})</h3>
          {review.length === 0 && <p className="muted">No conflicts to review.</p>}
          {review.map((prompt) => (
            <PromptCard key={prompt.id} prompt={prompt} state={state} onAnswer={onAnswer} onDismiss={() => undefined} />
          ))}
        </div>
      </div>
    </div>
  );
}
