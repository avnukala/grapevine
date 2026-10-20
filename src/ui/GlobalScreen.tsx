import { GitMerge, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "../api";
import type { WeightFactors } from "../state/derive";
import type { Prompt, Run, Session } from "../state/types";
import { useRun } from "../state/useRun";
import { GraphView } from "./GraphView";
import { Stats, ViewSwitch } from "./LiveScreen";
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
    try {
      setMergeRun(await api.merge());
      toast.success("Merged the accepted runs");
    } catch (e) {
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  const onAnswer = (prompt: Prompt, answer: Answer) =>
    api.resolveMerge(mergeRun!.id, prompt.id, answer as { action: string; fact_id?: number }).catch((e) => toast.error(String(e)));

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-testid="global">
      <div className="flex flex-wrap items-center gap-2 border-b border-border bg-card/30 px-4 py-2">
        <Button size="sm" disabled={busy || accepted.length === 0} onClick={() => void runMerge()}>
          <GitMerge /> {busy ? "Merging…" : "Merge accepted runs"}
        </Button>
        <span className="text-xs text-muted-foreground">Accepted:</span>
        {accepted.map(({ session, run }) => (
          <Badge key={run.id} variant="secondary">
            {session.name} · run {run.id}
          </Badge>
        ))}
        {accepted.length === 0 && <span className="text-xs text-muted-foreground">none yet</span>}
        {mergeRun && (
          <Badge variant="outline" className="ml-auto">
            merge run {mergeRun.id}
          </Badge>
        )}
      </div>
      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">
          <div className="absolute inset-0">
            {view === "graph" ? (
              <GraphView state={state} pulses={[]} speakingId={null} factors={factors} selectedId={selectedId} onSelect={setSelectedId} />
            ) : (
              <TimelineView state={state} recentFactIds={[]} recentEventIds={[]} />
            )}
          </div>
          <div className="absolute top-4 left-4 z-10 flex flex-col gap-3">
            <ViewSwitch view={view} onView={setView} />
            {view === "graph" && <Stats state={state} />}
          </div>
        </div>
        <ScrollArea className="w-[400px] shrink-0 border-l border-border bg-card/30">
          <div className="space-y-3 p-4">
            <h3 className="flex items-center gap-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              <ShieldCheck className="size-4" /> Review ({review.length})
            </h3>
            {review.length === 0 && <p className="text-sm text-muted-foreground">No conflicts to review.</p>}
            {review.map((prompt) => (
              <PromptCard key={prompt.id} prompt={prompt} state={state} onAnswer={onAnswer} onDismiss={() => undefined} />
            ))}
          </div>
        </ScrollArea>
      </div>
      <PersonDetails state={state} personId={selectedId} onClose={() => setSelectedId(null)} />
    </div>
  );
}
