import { CalendarRange, Check, Network, Send } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import { api } from "../api";
import { CATEGORIES, CATEGORY_LIST } from "../state/categories";
import { liveFacts, personName, roomIds } from "../state/derive";
import type { WeightFactors } from "../state/derive";
import { recent } from "../state/pulses";
import type { Phase, Prompt, Run, RunState, Segment, Session } from "../state/types";
import { useNow, useRun } from "../state/useRun";
import type { TimedChange } from "../state/useRun";
import { GraphView } from "./GraphView";
import { Onboarding } from "./Onboarding";
import { PersonDetails } from "./PersonDetails";
import type { Answer } from "./PromptCard";
import { PromptPanel } from "./PromptPanel";
import { SessionControls } from "./SessionControls";
import { TimelineView } from "./TimelineView";
import { Transcript } from "./Transcript";
import { WeightPanel } from "./WeightPanel";

export type View = "graph" | "timeline";
const GLOW_MS = 2500;

interface ViewSwitchProps {
  view: View;
  onView: (view: View) => void;
}

export function ViewSwitch({ view, onView }: ViewSwitchProps) {
  return (
    <ToggleGroup
      type="single"
      variant="outline"
      size="sm"
      value={view}
      onValueChange={(value) => value && onView(value as View)}
      className="bg-card/80 backdrop-blur"
    >
      <ToggleGroupItem value="graph" aria-label="Graph">
        <Network /> Graph
      </ToggleGroupItem>
      <ToggleGroupItem value="timeline" aria-label="Timeline">
        <CalendarRange /> Timeline
      </ToggleGroupItem>
    </ToggleGroup>
  );
}

interface StatsProps {
  state: RunState;
}

export function Stats({ state }: StatsProps) {
  const counts: [string, number][] = [
    ["people", Object.values(state.people).filter((p) => p.merged_into_id === null).length],
    ["relationships", Object.keys(state.relationships).length],
    ["facts", liveFacts(state).length],
    ["events", Object.keys(state.events).length],
  ];
  return (
    <div className="flex gap-1.5" data-testid="stats">
      {counts.map(([label, count]) => (
        <Badge key={label} variant="secondary" className="bg-card/80 backdrop-blur">
          <span className="font-semibold tabular-nums">{count}</span>
          <span className="text-muted-foreground">{label}</span>
        </Badge>
      ))}
    </div>
  );
}

const PHASE_LABELS: Record<Phase, string> = { intro: "Introductions", conversation: "Conversation", ended: "Ended" };

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
  const [onboarding, setOnboarding] = useState(true);
  const live = useRun(runId);
  const now = useNow();
  const { state } = live;

  const [going, setGoing] = useState<Run | null>(null);

  const refresh = useCallback(async () => {
    const [all, current] = await Promise.all([api.sessions(), api.activeRun()]);
    setSessions(all);
    setGoing(current);
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh, live.status]);

  const open = (run: Run) => {
    setOnboarding(false);
    setSessionId(run.session_id);
    setRunId(run.id);
  };
  useEffect(() => {
    void api.activeRun().then((run) => run && open(run));
  }, []);
  useEffect(() => {
    if (live.error) toast.error(live.error);
  }, [live.error]);

  const session = sessions.find((s) => s.id === sessionId) ?? null;
  const run = session?.runs.find((r) => r.id === runId);
  const status = live.status ?? run?.status ?? null;
  const runActive = status === "running";
  const threshold = Number(run?.config?.voice_threshold ?? 0.86);

  const changed = useMemo(() => recent(live.changes, (c) => (c as TimedChange).receivedAt, now), [live.changes, now]);
  const lastSegment = Object.values(state.segments).sort((x, y) => y.id - x.id)[0];
  const lastSegmentAt = live.changes.find((c) => c.op.type === "segment" && c.op.segment.id === lastSegment?.id)?.receivedAt;
  const speakingId =
    live.speaking.personId ?? (lastSegment && lastSegmentAt && now - lastSegmentAt < GLOW_MS ? lastSegment.person_id : null);

  const guard = (task: () => Promise<unknown>) => task().catch((e) => toast.error(String(e)));
  const onAnswer = (prompt: Prompt, answer: Answer) => guard(() => api.answer(runId!, prompt.id, answer as never));
  const onDismiss = (prompt: Prompt) => guard(() => api.dismiss(runId!, prompt.id));
  const onAssign = (segment: Segment, answer: { person_id: number } | { new_name: string }) =>
    guard(() => api.assignSpeaker(runId!, segment.id, answer));

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <SessionControls
        sessions={sessions}
        sessionId={sessionId}
        runId={runId}
        runActive={runActive}
        phase={state.phase}
        onSession={(id) => {
          setOnboarding(false);
          setSessionId(id);
          setRunId(sessions.find((s) => s.id === id)?.runs[0]?.id ?? null);
        }}
        onRun={setRunId}
        onRefresh={refresh}
        onOnboard={() => setOnboarding(true)}
      />

      {going && going.id !== runId && (
        <div className="flex items-center gap-3 border-b border-red-500/30 bg-red-500/10 px-4 py-2 text-sm" data-testid="going">
          <span className="size-2 animate-pulse rounded-full bg-red-500" />
          Run {going.id} ({going.kind}) is still going. Only one run can go at a time.
          <Button size="sm" variant="secondary" onClick={() => open(going)}>
            Open it
          </Button>
          <Button size="sm" variant="ghost" onClick={() => guard(() => api.stop(going.id).then(refresh))}>
            Stop it
          </Button>
        </div>
      )}

      {onboarding ? (
        <Onboarding
          blocked={going !== null}
          onCreated={async (id, newRunId) => {
            await refresh();
            setSessionId(id);
            setRunId(newRunId);
            setOnboarding(false);
          }}
          onCancel={sessionId !== null ? () => setOnboarding(false) : null}
        />
      ) : (
        <div className="relative min-h-0 flex-1">
          <div className="absolute inset-0">
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
          </div>

          <div className="pointer-events-none absolute top-4 left-4 z-10 flex flex-col gap-3 [&>*]:pointer-events-auto">
            <div className="flex items-center gap-2">
              <ViewSwitch view={view} onView={setView} />
              <Badge variant="outline" className="gap-1.5 bg-card/80 backdrop-blur" data-testid="run-status">
                {runActive && <span className="size-2 animate-pulse rounded-full bg-red-500" />}
                {runId ? `Run ${runId} · ${status ?? ""} · ${PHASE_LABELS[state.phase]}` : "No run"}
              </Badge>
            </div>
            {view === "graph" && <Stats state={state} />}
            {state.phase === "intro" && runId !== null && (
              <div className="w-72 rounded-xl border border-emerald-400/30 bg-emerald-400/10 p-3 text-sm backdrop-blur" data-testid="intro">
                <div className="font-semibold text-emerald-700 dark:text-emerald-300">Introductions</div>
                <p className="mt-1 text-muted-foreground">Each person says "I'm" and their name, then a few sentences.</p>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {roomIds(state).map((id) => {
                    const heard = Object.values(state.speakers).includes(id);
                    return (
                      <Badge key={id} variant={heard ? "default" : "outline"} className={heard ? "" : "text-muted-foreground"}>
                        {heard && <Check />}
                        {personName(state, id)}
                      </Badge>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          <PromptPanel
            state={state}
            className={view === "timeline" ? "top-auto right-[380px] bottom-16" : undefined}
            onAnswer={onAnswer}
            onDismiss={onDismiss}
          />

          {view === "graph" && (
            <div className="pointer-events-none absolute inset-x-4 bottom-4 z-10 flex items-end gap-4">
              <div
                className={cn(
                  "pointer-events-auto max-w-3xl flex-1 rounded-xl border border-border bg-card/80 p-3 shadow-2xl backdrop-blur",
                  Object.keys(state.segments).length === 0 && !runActive && "hidden",
                )}
              >
                <Transcript state={state} voiceThreshold={threshold} limit={3} partial={live.partial} onAssign={onAssign} />
                {runActive && <TypedTurn runId={runId!} />}
              </div>
              <div className="pointer-events-auto ml-auto flex flex-col items-end gap-2">
                <WeightPanel factors={factors} onChange={onFactors} />
                <div className="flex gap-3 rounded-lg border border-border bg-card/80 px-3 py-1.5 text-[11px] text-muted-foreground backdrop-blur">
                  {CATEGORY_LIST.map((c) => (
                    <span key={c} className="flex items-center gap-1.5">
                      <span className="h-0.5 w-3 rounded" style={{ background: CATEGORIES[c].color }} />
                      {c}
                    </span>
                  ))}
                  <span className="flex items-center gap-1.5">
                    <span className="h-0 w-3 border-t border-dashed border-muted-foreground" /> past
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <PersonDetails state={state} personId={selectedId} onClose={() => setSelectedId(null)} />
    </div>
  );
}

interface TypedTurnProps {
  runId: number;
}

function TypedTurn({ runId }: TypedTurnProps) {
  const [speaker, setSpeaker] = useState("");
  const [text, setText] = useState("");
  return (
    <form
      className="mt-2 flex gap-2 border-t border-border pt-2"
      onSubmit={(e) => {
        e.preventDefault();
        if (!speaker.trim() || !text.trim()) return;
        api
          .text(runId, speaker, text)
          .then(() => setText(""))
          .catch((err) => toast.error(String(err)));
      }}
    >
      <Input className="w-36" value={speaker} onChange={(e) => setSpeaker(e.target.value)} placeholder="Speaker" aria-label="Speaker" />
      <Input value={text} onChange={(e) => setText(e.target.value)} placeholder="Type what they said" aria-label="Turn" />
      <Button size="icon" aria-label="Add turn">
        <Send />
      </Button>
    </form>
  );
}
