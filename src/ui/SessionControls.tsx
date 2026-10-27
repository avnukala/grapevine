import { Check, FastForward, FileAudio, Keyboard, LoaderCircle, Mic, Play, Plus, Square, Upload, UserCheck } from "lucide-react";
import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { api } from "../api";
import type { Source } from "../api";
import type { Phase, Session } from "../state/types";

interface SessionControlsProps {
  sessions: Session[];
  sessionId: number | null;
  runId: number | null;
  runActive: boolean;
  phase: Phase;
  onSession: (id: number | null) => void;
  onRun: (id: number | null) => void;
  onRefresh: () => Promise<void>;
  onOnboard: () => void;
}

export function SessionControls(props: SessionControlsProps) {
  const { sessions, sessionId, runId, runActive, phase, onSession, onRun, onRefresh, onOnboard } = props;
  const [files, setFiles] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  // Starting a run loads the speech models first, which takes a few seconds.
  const [starting, setStarting] = useState<string | null>(null);
  // A stopped run still reads its last turns and writes the review, which takes about a minute.
  const [stoppedId, setStoppedId] = useState<number | null>(null);
  const finishing = phase === "review" || (runId !== null && stoppedId === runId);
  const session = sessions.find((s) => s.id === sessionId) ?? null;
  const run = session?.runs.find((r) => r.id === runId) ?? null;

  useEffect(() => {
    api.audioFiles().then(setFiles).catch(() => setFiles([]));
  }, [sessions.length]);

  async function act(task: () => Promise<unknown>) {
    setBusy(true);
    try {
      await task();
      await onRefresh();
    } catch (e) {
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  const start = (source: Source, speed: number | null = null) => {
    setStarting(`${source}-${speed}`);
    void act(async () => onRun((await api.startRun({ session_id: sessionId!, source, speed })).id)).finally(() =>
      setStarting(null),
    );
  };
  const icon = (key: string, idle: ReactNode) =>
    starting === key ? <LoaderCircle className="animate-spin" /> : idle;

  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-border bg-card/30 px-4 py-2" data-testid="controls">
      <Select value={sessionId ? String(sessionId) : undefined} onValueChange={(v) => onSession(Number(v))}>
        <SelectTrigger className="w-72" aria-label="Session" data-testid="session-select">
          <SelectValue placeholder="Pick a session" />
        </SelectTrigger>
        <SelectContent>
          {sessions.map((s) => (
            <SelectItem key={s.id} value={String(s.id)}>
              {s.name} <span className="text-muted-foreground">{s.recorded_at}</span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {session && (
        <Select value={runId ? String(runId) : undefined} onValueChange={(v) => onRun(Number(v))}>
          <SelectTrigger className="w-56" aria-label="Run" data-testid="run-select">
            <SelectValue placeholder="No runs yet" />
          </SelectTrigger>
          <SelectContent>
            {session.runs.map((r) => (
              <SelectItem key={r.id} value={String(r.id)}>
                Run {r.id} · {r.kind} · {r.status}
                {r.accepted ? " · accepted" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {run?.accepted ? (
        <Badge variant="secondary" className="gap-1">
          <Check className="size-3" /> accepted
        </Badge>
      ) : null}

      <Separator orientation="vertical" className="mx-1 h-6" />

      {session && !runActive && (
        <>
          <Button size="sm" disabled={busy} onClick={() => start("mic")}>
            {icon("mic-null", <Mic />)} Start live
          </Button>
          {session.audio_uri && (
            <>
              <Button size="sm" variant="secondary" disabled={busy} onClick={() => start("file", 1)}>
                {icon("file-1", <Play />)} Replay 1×
              </Button>
              <Button size="sm" variant="secondary" disabled={busy} onClick={() => start("file", null)}>
                {icon("file-null", <FastForward />)} Replay fast
              </Button>
            </>
          )}
          <Button size="sm" variant="outline" disabled={busy} onClick={() => start("text")}>
            {icon("text-null", <Keyboard />)} Typed run
          </Button>
          {run && run.kind !== "merge" && !run.accepted && (run.status === "done" || run.status === "stopped") && (
            <Button size="sm" variant="outline" disabled={busy} onClick={() => act(() => api.accept(run.id))}>
              <Check /> Accept run
            </Button>
          )}
        </>
      )}
      {runActive && runId !== null && (
        <>
          {phase === "intro" && !finishing && (
            <Button size="sm" disabled={busy} onClick={() => act(() => api.introDone(runId))}>
              <UserCheck /> Done with introductions
            </Button>
          )}
          {finishing ? (
            <Button size="sm" variant="outline" disabled data-testid="finishing">
              <LoaderCircle className="animate-spin" /> Finishing: writing the review
            </Button>
          ) : (
            <Button
              size="sm"
              variant="destructive"
              disabled={busy}
              onClick={() => {
                setStoppedId(runId);
                act(() => api.stop(runId));
              }}
            >
              <Square /> Stop
            </Button>
          )}
        </>
      )}

      <Popover>
        <PopoverTrigger asChild>
          <Button size="sm" variant="ghost" className="ml-auto">
            <Plus /> New session
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-96 space-y-4">
          <Button className="w-full" onClick={onOnboard}>
            <Mic /> New live session
          </Button>
          <label className="flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-border px-3 py-3 text-sm text-muted-foreground hover:bg-accent/40">
            <Upload className="size-4" /> Upload a recording
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void act(async () => onSession((await api.upload(file)).id));
              }}
            />
          </label>
          {files.length > 0 && (
            <div>
              <div className="mb-1.5 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Recordings</div>
              <div className="max-h-56 space-y-0.5 overflow-auto">
                {files.map((file) => (
                  <button
                    key={file}
                    className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-accent"
                    onClick={() =>
                      act(async () => onSession((await api.newSession({ name: file.split("/").pop()!, audio_uri: file })).id))
                    }
                  >
                    <FileAudio className="size-4 text-muted-foreground" />
                    {file.split("/").pop()}
                  </button>
                ))}
              </div>
            </div>
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
