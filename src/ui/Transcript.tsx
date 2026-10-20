import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { GENDER_COLORS, initials } from "../state/categories";
import { personName, roomIds } from "../state/derive";
import type { RunState, Segment } from "../state/types";
import type { Partial } from "../state/useRun";

interface TranscriptProps {
  state: RunState;
  voiceThreshold: number;
  limit?: number;
  currentMs?: number | null;
  showTimings?: boolean;
  partial?: Partial | null;
  onAssign?: (segment: Segment, answer: { person_id: number } | { new_name: string }) => void;
  onSeek?: (ms: number) => void;
}

export function isUnknown(segment: Segment, threshold: number): boolean {
  return segment.person_id === null || (segment.speaker_confidence !== null && segment.speaker_confidence < threshold);
}

function clock(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function Transcript(props: TranscriptProps) {
  const { state, voiceThreshold, limit, currentMs, showTimings, partial, onAssign, onSeek } = props;
  const segments = Object.values(state.segments).sort((x, y) => x.start_ms - y.start_ms);
  const shown = limit ? segments.slice(-(partial ? limit - 1 : limit)) : segments;
  return (
    <ol className="space-y-1.5" data-testid="transcript">
      {shown.map((segment) => {
        const naming = state.phase === "intro" && segment.person_id === null && segment.speaker_confidence === null;
        const unknown = !naming && isUnknown(segment, voiceThreshold);
        const current = currentMs != null && segment.start_ms <= currentMs && currentMs < segment.end_ms;
        const name = naming
          ? `Introducing (${segment.speaker_label})`
          : unknown
            ? `Unknown (${segment.speaker_label})`
            : personName(state, segment.person_id);
        const gender = (segment.person_id !== null && state.people[segment.person_id]?.gender) || "unknown";
        return (
          <li
            key={segment.id}
            className={cn(
              "flex items-start gap-3 rounded-lg px-2 py-1.5 transition",
              current && "bg-amber-300/10 ring-1 ring-amber-300/40",
              unknown && "bg-orange-500/5",
            )}
          >
            <button
              className="mt-0.5 w-9 shrink-0 text-right font-mono text-[11px] text-muted-foreground tabular-nums hover:text-foreground"
              onClick={() => onSeek?.(segment.start_ms)}
              disabled={!onSeek}
            >
              {clock(segment.start_ms)}
            </button>
            <span
              className={cn(
                "flex size-6 shrink-0 items-center justify-center rounded-full text-[9px] font-bold text-white",
                unknown && "ring-2 ring-orange-400",
              )}
              style={{ background: unknown ? "#9a3412" : GENDER_COLORS[gender][1] }}
            >
              {unknown ? "?" : naming ? "…" : initials(name)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-xs">
                <span className={cn("font-semibold", unknown && "text-orange-700 dark:text-orange-300")}>{name}</span>
                {segment.speaker_confidence !== null && (
                  <span className="text-muted-foreground tabular-nums">voice {Math.round(segment.speaker_confidence * 100)}%</span>
                )}
              </div>
              <p className="text-sm leading-snug">{segment.text}</p>
              {unknown && onAssign && <WhoSpoke state={state} onPick={(answer) => onAssign(segment, answer)} />}
              {showTimings && Object.keys(segment.timings).length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {Object.entries(segment.timings).map(([step, ms]) => (
                    <Badge key={step} variant="outline" className="font-mono text-[10px] text-muted-foreground">
                      {step.replace("_ms", "")} {ms}ms
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </li>
        );
      })}
      {partial && <PartialLine state={state} partial={partial} />}
    </ol>
  );
}

interface PartialLineProps {
  state: RunState;
  partial: Partial;
}

function PartialLine({ state, partial }: PartialLineProps) {
  const name = partial.personId === null ? `Someone (${partial.label})` : personName(state, partial.personId);
  return (
    <li className="flex items-start gap-3 rounded-lg px-2 py-1.5" data-testid="partial">
      <span className="mt-0.5 w-9 shrink-0 text-right font-mono text-[11px] text-muted-foreground tabular-nums">
        {clock(partial.startMs)}
      </span>
      <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-emerald-500 text-[9px] font-bold text-white">
        {partial.personId === null ? "…" : initials(name)}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-semibold">{name}</span>
          <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300">
            <span className="size-1.5 animate-pulse rounded-full bg-emerald-500" /> speaking
          </span>
        </div>
        <p className="text-sm leading-snug text-muted-foreground">
          {partial.text}
          <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-muted-foreground align-middle" />
        </p>
      </div>
    </li>
  );
}

interface WhoSpokeProps {
  state: RunState;
  onPick: (answer: { person_id: number } | { new_name: string }) => void;
}

function WhoSpoke({ state, onPick }: WhoSpokeProps) {
  const [name, setName] = useState("");
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-orange-700 dark:text-orange-300">
      Who spoke?
      {roomIds(state).map((id) => (
        <Button key={id} size="xs" variant="secondary" onClick={() => onPick({ person_id: id })}>
          {personName(state, id).split(" ")[0]}
        </Button>
      ))}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) onPick({ new_name: name.trim() });
        }}
      >
        <Input className="h-6 w-28 text-xs" value={name} onChange={(e) => setName(e.target.value)} placeholder="New name" />
      </form>
    </div>
  );
}
