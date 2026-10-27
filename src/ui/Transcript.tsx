import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { GENDER_COLORS, KIND_COLORS, initials } from "../state/categories";
import { personName, roomIds } from "../state/derive";
import type { RunState, Segment } from "../state/types";
import { parsePartialJson } from "../state/partialJson";
import type { Preview, Thinking } from "../state/useRun";

interface TranscriptProps {
  state: RunState;
  voiceThreshold: number;
  limit?: number;
  currentMs?: number | null;
  showTimings?: boolean;
  partial?: Preview | null;
  extracting?: Thinking | null;
  onAssign?: (segment: Segment, answer: SpeakerPick) => void;
  onSeek?: (ms: number) => void;
}

export type SpeakerPick = ({ person_id: number } | { new_name: string }) & { every_line?: boolean };

export function isUnknown(segment: Segment, threshold: number): boolean {
  if (segment.speaker_manual) return false;
  return segment.person_id === null || (segment.speaker_confidence !== null && segment.speaker_confidence < threshold);
}

export function clock(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function Transcript(props: TranscriptProps) {
  const { state, voiceThreshold, limit, currentMs, showTimings, partial, extracting, onAssign, onSeek } = props;
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
              current && "bg-warn/10 ring-1 ring-warn/40",
              unknown && "bg-alert/5",
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
                "flex size-6 shrink-0 items-center justify-center rounded-full text-[9px] font-bold text-ink",
                unknown && "ring-2 ring-alert",
              )}
              style={{ background: unknown ? "var(--alert)" : GENDER_COLORS[gender] }}
            >
              {unknown ? "?" : naming ? "…" : initials(name)}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-xs">
                {onAssign ? (
                  <SpeakerPicker state={state} segment={segment} onPick={(answer) => onAssign(segment, answer)}>
                    <span className={cn("font-semibold", unknown && "text-alert")}>{name}</span>
                  </SpeakerPicker>
                ) : (
                  <span className={cn("font-semibold", unknown && "text-alert")}>{name}</span>
                )}
                {segment.speaker_manual ? (
                  <span className="text-muted-foreground">set by hand</span>
                ) : (
                  segment.speaker_confidence !== null && (
                    <span className="text-muted-foreground tabular-nums">
                      voice {Math.round(segment.speaker_confidence * 100)}%
                    </span>
                  )
                )}
              </div>
              <p className="text-sm leading-snug">{segment.text}</p>
              {extracting?.segmentId === segment.id ? (
                <ExtractedList items={extractedSoFar(extracting.text, name)} live />
              ) : (
                <ExtractedList items={factsOf(state, segment.id)} live={false} />
              )}
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
      {partial && <PreviewLine state={state} partial={partial} />}
    </ol>
  );
}

interface PreviewLineProps {
  state: RunState;
  partial: Preview;
}

function PreviewLine({ state, partial }: PreviewLineProps) {
  const name = partial.personId !== null ? personName(state, partial.personId) : "Someone";
  return (
    <li className="flex items-start gap-3 rounded-lg px-2 py-1.5" data-testid="partial">
      <span className="mt-0.5 w-9 shrink-0 text-right font-mono text-[11px] text-muted-foreground tabular-nums">
        {clock(partial.startMs)}
      </span>
      <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-speak text-[9px] font-bold text-ink">
        {partial.personId === null ? "…" : initials(name)}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-xs">
          <span className="font-semibold">{name}</span>
          <span className="flex items-center gap-1 text-speak">
            <span className="size-1.5 animate-pulse rounded-full bg-speak" /> speaking
          </span>
        </div>
        <p className="text-sm leading-snug text-muted-foreground">
          {partial.text.split(" ").map((word, i) => (
            <span key={i} className="animate-in duration-300 fade-in">
              {word}{" "}
            </span>
          ))}
          <span className="ml-0.5 inline-block h-3.5 w-0.5 animate-pulse bg-muted-foreground align-middle" />
        </p>
      </div>
    </li>
  );
}

interface Extracted {
  kind: keyof typeof KIND_COLORS | "event";
  text: string;
}

// What the model is pulling out of a turn while it writes, from its unfinished JSON.
function extractedSoFar(text: string, speaker: string): Extracted[] {
  const output = parsePartialJson(text) as {
    facts?: { kind?: Extracted["kind"]; text?: string }[];
    events?: { name?: string; start_date?: string | null }[];
    relationships?: { a?: string; b?: string; label?: string }[];
  };
  const who = (ref?: string) => (ref === "speaker" ? speaker : (ref ?? "…"));
  return [
    ...(output?.facts ?? []).filter((f) => f.text).map((f) => ({ kind: f.kind ?? "personal", text: f.text! })),
    ...(output?.events ?? [])
      .filter((e) => e.name)
      .map((e) => ({ kind: "event" as const, text: `${e.name}${e.start_date ? ` (${e.start_date})` : ""}` })),
    ...(output?.relationships ?? [])
      .filter((r) => r.label)
      .map((r) => ({ kind: "relationship" as const, text: `${who(r.a)} and ${who(r.b)}: ${r.label}` })),
  ];
}

function factsOf(state: RunState, segmentId: number): Extracted[] {
  return Object.values(state.facts)
    .filter((f) => f.segment_ids.includes(segmentId) && f.revised_by_fact_id === null)
    .map((f) => ({ kind: f.kind, text: f.text }));
}

interface ExtractedListProps {
  items: Extracted[];
  live: boolean;
}

function ExtractedList({ items, live }: ExtractedListProps) {
  if (items.length === 0 && !live) return null;
  return (
    <ul className="mt-1 space-y-0.5 text-xs text-muted-foreground" data-testid={live ? "extracting" : "extracted"}>
      {items.map((item, i) => (
        <li key={i} className="flex items-baseline gap-1.5 animate-in duration-300 fade-in">
          <span
            className="size-1.5 shrink-0 translate-y-[-1px] rounded-full"
            style={{ background: item.kind === "event" ? "var(--primary)" : KIND_COLORS[item.kind] }}
          />
          {item.text}
        </li>
      ))}
      {live && (
        <li className="flex items-center gap-1.5 text-primary">
          <span className="size-1.5 animate-pulse rounded-full bg-primary" /> reading
        </li>
      )}
    </ul>
  );
}

interface SpeakerPickerProps {
  state: RunState;
  segment: Segment;
  onPick: (answer: SpeakerPick) => void;
  children: React.ReactNode;
}

// Fixing one line of an introduction usually means the whole voice was named wrong, so that is the default.
function SpeakerPicker({ state, segment, onPick, children }: SpeakerPickerProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [everyLine, setEveryLine] = useState(true);
  const room = roomIds(state);
  const others = Object.keys(state.people)
    .map(Number)
    .filter((id) => !room.includes(id));
  const pick = (answer: SpeakerPick) => {
    onPick({ ...answer, every_line: everyLine });
    setOpen(false);
    setName("");
  };
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button className="rounded hover:underline" aria-label="Change speaker" data-testid="speaker-picker">
          {children}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-72 space-y-3">
        <p className="text-sm font-medium">Who said this?</p>
        <div className="flex flex-wrap gap-1.5">
          {[...room, ...others].map((id) => (
            <Button
              key={id}
              size="xs"
              variant={id === segment.person_id ? "default" : room.includes(id) ? "secondary" : "outline"}
              onClick={() => pick({ person_id: id })}
            >
              {personName(state, id)}
            </Button>
          ))}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (name.trim()) pick({ new_name: name.trim() });
          }}
        >
          <Input className="h-7 text-xs" value={name} onChange={(e) => setName(e.target.value)} placeholder="New name" />
        </form>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <input type="checkbox" checked={everyLine} onChange={(e) => setEveryLine(e.target.checked)} />
          Every line from this voice ({segment.speaker_label})
        </label>
      </PopoverContent>
    </Popover>
  );
}

interface WhoSpokeProps {
  state: RunState;
  onPick: (answer: { person_id: number } | { new_name: string }) => void;
}

function WhoSpoke({ state, onPick }: WhoSpokeProps) {
  const [name, setName] = useState("");
  return (
    <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-alert">
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
