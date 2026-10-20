import { BookOpen, ScanSearch } from "lucide-react";
import { parsePartialJson } from "../state/partialJson";
import type { Thinking } from "../state/useRun";

interface Line {
  title: string;
  text: string;
}

// The lines of an end-of-session pass so far: relationships and groups from the review, stories from the summary.
function linesSoFar(thinking: Thinking): Line[] {
  const output = parsePartialJson(thinking.text) as
    | {
        stories?: { title?: string; text?: string }[];
        relationships?: { a?: string; b?: string; label?: string; summary?: string }[];
        groups?: { name?: string; members?: string[] }[];
      }
    | undefined;
  if (thinking.worker === "summarizer") {
    return (output?.stories ?? []).filter((s) => s.title).map((s) => ({ title: s.title!, text: s.text ?? "" }));
  }
  return [
    ...(output?.relationships ?? [])
      .filter((r) => r.b && r.label)
      .map((r) => ({ title: `${r.a} and ${r.b}: ${r.label}`, text: r.summary ?? "" })),
    ...(output?.groups ?? [])
      .filter((g) => g.name)
      .map((g) => ({ title: g.name!, text: (g.members ?? []).join(", ") })),
  ];
}

interface StoryWriterProps {
  thinking: Thinking;
}

export function StoryWriter({ thinking }: StoryWriterProps) {
  const reviewing = thinking.worker === "reviewer";
  const Icon = reviewing ? ScanSearch : BookOpen;
  return (
    <div
      className="pointer-events-auto absolute right-4 bottom-24 z-20 max-h-[60%] w-96 overflow-auto rounded-xl border border-border bg-card p-4"
      data-testid="story-writer"
    >
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wider text-primary uppercase">
        <Icon className="size-4" /> {reviewing ? "Reading the whole conversation" : "Writing stories"}
        <span className="size-1.5 animate-pulse rounded-full bg-primary" />
      </div>
      <ol className="space-y-2.5">
        {linesSoFar(thinking).map((line, i) => (
          <li key={i} className="animate-in duration-300 fade-in">
            <div className="text-sm font-semibold">{line.title}</div>
            {line.text && <p className="text-sm leading-snug text-muted-foreground">{line.text}</p>}
          </li>
        ))}
      </ol>
    </div>
  );
}
