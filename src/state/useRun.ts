import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { applyOp, emptyState } from "./reducer";
import type { Change, RunState } from "./types";

export type ServerMessage =
  | { type: "change"; run_id: number; change: Change }
  | { type: "activity"; run_id: number; speaker_label: string | null; person_id: number | null }
  | { type: "status"; run_id: number; status: string }
  | { type: "partial"; run_id: number; start_ms: number; text: null }
  | {
      type: "partial";
      run_id: number;
      start_ms: number;
      text: string;
      speaker_label: string | null;
      person_id: number | null;
    }
  | { type: "thinking"; run_id: number; worker: Worker; segment_id?: number; text: string | null }
  | { type: "connected" };

const listeners = new Set<(message: ServerMessage) => void>();
let socket: WebSocket | null = null;

function connect(): void {
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onopen = () => listeners.forEach((listen) => listen({ type: "connected" }));
  socket.onmessage = (event) => {
    const message = JSON.parse(event.data) as ServerMessage;
    listeners.forEach((listen) => listen(message));
  };
  socket.onclose = () => setTimeout(connect, 1000);
}

export function subscribe(listen: (message: ServerMessage) => void): () => void {
  if (!socket) connect();
  listeners.add(listen);
  return () => listeners.delete(listen);
}

export interface TimedChange extends Change {
  receivedAt: number;
}

// The words of a turn that is still going. The server sends it only for runs at real time, and never saves it.
export interface Preview {
  startMs: number;
  label: string | null;
  personId: number | null;
  text: string;
}

export type Worker = "extractor" | "prompter" | "reviewer" | "summarizer";

// The output of a model call that is still going, as the model writes it.
export interface Thinking {
  worker: Worker;
  segmentId: number | null;
  text: string;
}

export interface LiveRun {
  changes: TimedChange[];
  state: RunState;
  speaking: { label: string | null; personId: number | null };
  partial: Preview | null;
  thinking: Partial<Record<Worker, Thinking>>;
  status: string | null;
  error: string | null;
}

// Loads the changes of a run and keeps them current from the WebSocket.
export function useRun(runId: number | null): LiveRun {
  const [changes, setChanges] = useState<TimedChange[]>([]);
  const [speaking, setSpeaking] = useState<LiveRun["speaking"]>({ label: null, personId: null });
  const [partial, setPartial] = useState<Preview | null>(null);
  const [thinking, setThinking] = useState<LiveRun["thinking"]>({});
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setChanges([]);
    setSpeaking({ label: null, personId: null });
    setPartial(null);
    setThinking({});
    setStatus(null);
    setError(null);
    if (runId === null) return;

    let cancelled = false;
    const merge = (incoming: Change[], receivedAt: number) =>
      setChanges((current) => {
        const known = new Set(current.map((c) => c.seq));
        const fresh = incoming.filter((c) => !known.has(c.seq)).map((c) => ({ ...c, receivedAt }));
        if (fresh.length === 0) return current;
        return [...current, ...fresh].sort((x, y) => x.seq - y.seq);
      });
    const load = () =>
      Promise.all([api.changes(runId), api.run(runId)])
        .then(([loaded, run]) => {
          if (cancelled) return;
          merge(loaded, 0);
          setStatus(run.status);
        })
        .catch((e) => setError(String(e)));

    const unsubscribe = subscribe((message) => {
      if (message.type === "connected") return void load();
      if (message.run_id !== runId) return;
      if (message.type === "change") merge([message.change], Date.now());
      if (message.type === "activity") setSpeaking({ label: message.speaker_label, personId: message.person_id });
      if (message.type === "status") {
        setStatus(message.status);
        setPartial(null);
      }
      if (message.type === "thinking") {
        const { worker, text, segment_id: segmentId = null } = message;
        setThinking((current) => {
          const next = { ...current };
          if (text === null) delete next[worker];
          else next[worker] = { worker, segmentId, text };
          return next;
        });
      }
      if (message.type === "partial" && message.text === null) {
        setPartial((current) => (current && current.startMs <= message.start_ms ? null : current));
      }
      if (message.type === "partial" && message.text !== null) {
        const { start_ms: startMs, speaker_label: label, person_id: personId, text } = message;
        setPartial({ startMs, label, personId, text });
      }
    });
    void load();
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [runId]);

  const state = useMemo(() => changes.reduce((s, c) => applyOp(s, c.op), emptyState()), [changes]);
  return { changes, state, speaking, partial, thinking, status, error };
}

// Re-renders every `ms` so views can age out short-lived highlights.
export function useNow(ms = 500): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(timer);
  }, [ms]);
  return now;
}
