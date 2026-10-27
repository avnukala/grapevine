import { Gauge } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { api, audioUrl } from "../api";
import { personName } from "../state/derive";
import type { WeightFactors } from "../state/derive";
import { recent, speakerAt } from "../state/pulses";
import { stateAt } from "../state/reducer";
import type { Change, ModelCall, RunState, Session } from "../state/types";
import { useRun } from "../state/useRun";
import { GraphView } from "./GraphView";
import { Stats, ViewSwitch } from "./LiveScreen";
import type { View } from "./LiveScreen";
import { TimelineView } from "./TimelineView";
import { Transcript } from "./Transcript";

const SPEEDS = [0.5, 1, 1.5, 2, 4];

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
  const duration = Math.max((audio.current?.duration ?? 0) * 1000 || 0, ...live.changes.map((c) => c.audio_ms), 1000);
  const state = useMemo(() => stateAt(live.changes, ms), [live.changes, ms]);
  const changed = useMemo(() => recent(live.changes, (c) => c.audio_ms, ms), [live.changes, ms]);
  const timings = useMemo(() => latencySummary(Object.values(live.state.segments).map((s) => s.timings)), [live.state]);

  const seek = (to: number) => {
    setMs(to);
    if (audio.current) audio.current.currentTime = to / 1000;
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-testid="debug">
      <div className="flex flex-wrap items-center gap-3 border-b border-border bg-card/30 px-4 py-2">
        <Select value={runId ? String(runId) : ""} onValueChange={(v) => setRunId(Number(v))}>
          <SelectTrigger className="w-96" aria-label="Run" data-testid="run-select">
            <SelectValue placeholder="Pick a run" />
          </SelectTrigger>
          <SelectContent>
            {sessions.flatMap((s) =>
              s.runs.map((r) => (
                <SelectItem key={r.id} value={String(r.id)}>
                  {s.name} · run {r.id} · {r.kind} · {r.status}
                  {r.accepted ? " · accepted" : ""}
                </SelectItem>
              )),
            )}
          </SelectContent>
        </Select>
        {session?.audio_uri && (
          <audio
            ref={audio}
            src={audioUrl(session.id)}
            controls
            className="h-8"
            onTimeUpdate={(e) => setMs(Math.round(e.currentTarget.currentTime * 1000))}
            onLoadedMetadata={(e) => (e.currentTarget.playbackRate = speed)}
          />
        )}
        <div className="flex items-center gap-1">
          <Gauge className="size-4 text-muted-foreground" />
          {SPEEDS.map((s) => (
            <Button key={s} size="xs" variant={speed === s ? "default" : "ghost"} onClick={() => setSpeed(s)}>
              {s}×
            </Button>
          ))}
        </div>
        <Slider
          className="min-w-48 flex-1"
          min={0}
          max={Math.ceil(duration)}
          step={100}
          value={[Math.min(ms, duration)]}
          onValueChange={([value]) => seek(value)}
          aria-label="Audio time"
          data-testid="scrubber"
        />
        <span className="font-mono text-xs tabular-nums" data-testid="clock">
          {(ms / 1000).toFixed(1)}s / {(duration / 1000).toFixed(1)}s
        </span>
      </div>

      {run && (
        <div className="flex flex-wrap gap-1.5 border-b border-border px-4 py-1.5 text-xs">
          <Badge variant="outline">run {run.id}</Badge>
          <Badge variant="outline">{run.kind}</Badge>
          <Badge variant="outline">{live.status ?? run.status}</Badge>
          <Badge variant="outline">code {run.code_version}</Badge>
          {Object.entries(timings).map(([step, value]) => (
            <Badge key={step} variant="secondary" className="font-mono">
              {step.replace("_ms", "")} {value}
            </Badge>
          ))}
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <div className="absolute inset-0">
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
          <div className="absolute top-4 left-4 z-10 flex flex-col gap-3">
            <ViewSwitch view={view} onView={setView} />
            {view === "graph" && <Stats state={state} />}
          </div>
        </div>

        <Tabs defaultValue="transcript" className="flex w-[480px] shrink-0 flex-col border-l border-border bg-card/30">
          <TabsList className="m-2">
            <TabsTrigger value="transcript">Transcript</TabsTrigger>
            <TabsTrigger value="calls">Model calls ({calls.length})</TabsTrigger>
            <TabsTrigger value="changes">Changes ({live.changes.length})</TabsTrigger>
          </TabsList>
          <ScrollArea className="min-h-0 flex-1 px-2 pb-4">
            <TabsContent value="transcript">
              <Transcript
                state={live.state}
                voiceThreshold={Number(run?.config?.voice_threshold ?? 0.86)}
                currentMs={ms}
                showTimings
                onSeek={seek}
              />
            </TabsContent>
            <TabsContent value="calls">
              <ModelCalls calls={calls} onSeek={seek} />
            </TabsContent>
            <TabsContent value="changes">
              <ChangeLog changes={live.changes} currentMs={ms} state={live.state} onSeek={seek} />
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </div>
    </div>
  );
}

function latencySummary(timings: Record<string, number>[]): Record<string, string> {
  const steps = ["detect_ms", "transcribe_ms", "extract_ms", "total_ms"];
  return Object.fromEntries(
    steps.map((step) => {
      const values = timings.map((t) => t[step]).filter((v): v is number => v !== undefined).sort((x, y) => x - y);
      return [step, values.length ? `p50 ${values[Math.floor(values.length / 2)]}ms` : "n/a"];
    }),
  );
}

interface ModelCallsProps {
  calls: ModelCall[];
  onSeek: (ms: number) => void;
}

function ModelCalls({ calls, onSeek }: ModelCallsProps) {
  return (
    <ol className="space-y-1" data-testid="model-calls">
      {calls.map((call) => (
        <li key={call.id} className="rounded-lg border border-border bg-card">
          <details>
            <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-xs">
              <span className="font-mono text-muted-foreground">#{call.id}</span>
              <Badge variant="secondary">{call.worker}</Badge>
              <span className={cn("tabular-nums", call.latency_ms > 3000 && "text-warn")}>
                {call.cached ? "reused" : `${call.latency_ms}ms`}
              </span>
              <span className="text-muted-foreground">{call.prompt_version}</span>
              {call.audio_ms !== null && (
                <button className="ml-auto font-mono text-primary hover:underline" onClick={() => onSeek(call.audio_ms!)}>
                  @{(call.audio_ms / 1000).toFixed(1)}s
                </button>
              )}
            </summary>
            <div className="space-y-2 border-t border-border p-3 text-xs">
              <div className="text-muted-foreground">
                model {call.model} · file {call.model_file_hash} · seed {call.seed} · input {call.input_hash.slice(0, 12)}
              </div>
              <pre className="max-h-72 overflow-auto rounded-md bg-background p-2 whitespace-pre-wrap">
                {call.request.messages[1]?.content}
              </pre>
              <pre className="max-h-72 overflow-auto rounded-md bg-background p-2 whitespace-pre-wrap text-speak">
                {prettyJson(call.response)}
              </pre>
            </div>
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
  state: RunState;
  onSeek: (ms: number) => void;
}

function ChangeLog({ changes, currentMs, state, onSeek }: ChangeLogProps) {
  return (
    <ol className="space-y-0.5 font-mono text-[11px]" data-testid="change-log">
      {changes.map((change) => (
        <li key={change.seq} className={cn("flex gap-2 px-1 py-0.5", change.audio_ms > currentMs && "opacity-35")}>
          <button className="w-12 shrink-0 text-right text-primary hover:underline" onClick={() => onSeek(change.audio_ms)}>
            {(change.audio_ms / 1000).toFixed(1)}s
          </button>
          <span className="w-9 shrink-0 text-muted-foreground">#{change.seq}</span>
          <span className="w-20 shrink-0 text-warn">{change.op.type}</span>
          <span className="min-w-0 truncate" title={describe(change, state)}>
            {describe(change, state)}
          </span>
        </li>
      ))}
    </ol>
  );
}

function describe(change: Change, state: RunState): string {
  const op = change.op;
  switch (op.type) {
    case "person":
      return `${op.person.name}${op.person.merged_into_id ? " merged" : ""}`;
    case "speaker":
      return `voice ${op.speaker.label} is ${personName(state, op.speaker.person_id)}`;
    case "segment":
      return `${op.segment.speaker_label}: ${op.segment.text}`;
    case "fact":
      return `[${op.fact.kind}] ${op.fact.text}${op.fact.disputed ? " (disputed)" : ""}${op.fact.revised_by_fact_id ? " (revised)" : ""}`;
    case "event":
      return `${op.event.name} ${op.event.start_date ?? ""}`;
    case "group":
      return op.group.name;
    case "member":
      return `${personName(state, op.member.person_id)} in ${state.groups[op.member.group_id]?.name ?? op.member.group_id}`;
    case "relationship":
      return `${personName(state, op.relationship.a)} - ${personName(state, op.relationship.b)}: ${op.relationship.category} "${op.relationship.label}" ${op.relationship.status}`;
    case "relationship_removed":
      return `relationship ${op.id} removed`;
    case "person_removed":
      return `person ${op.id} removed`;
    case "prompt":
      return `[${op.prompt.reason}] ${op.prompt.text}${op.prompt.dismissed_at_ms !== null ? " (dismissed)" : ""}`;
    case "phase":
      return op.phase;
    case "roster":
      return `roster ${op.person_ids.map((id) => personName(state, id)).join(", ")}`;
  }
}
