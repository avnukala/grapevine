import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { applyOp, emptyState } from "./reducer";
import type { Change, RunState } from "./types";

export type ServerMessage =
  | { type: "change"; run_id: number; change: Change }
  | { type: "activity"; run_id: number; speaker_label: string | null; person_id: number | null }
  | { type: "status"; run_id: number; status: string }
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

export interface LiveRun {
  changes: TimedChange[];
  state: RunState;
  speaking: { label: string | null; personId: number | null };
  status: string | null;
  error: string | null;
}

// Loads the changes of a run and keeps them current from the WebSocket.
export function useRun(runId: number | null): LiveRun {
  const [changes, setChanges] = useState<TimedChange[]>([]);
  const [speaking, setSpeaking] = useState<LiveRun["speaking"]>({ label: null, personId: null });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setChanges([]);
    setSpeaking({ label: null, personId: null });
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
      if (message.type === "status") setStatus(message.status);
    });
    void load();
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [runId]);

  const state = useMemo(() => changes.reduce((s, c) => applyOp(s, c.op), emptyState()), [changes]);
  return { changes, state, speaking, status, error };
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
