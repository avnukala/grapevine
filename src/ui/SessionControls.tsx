import { useEffect, useState } from "react";
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
  onError: (message: string | null) => void;
}

export function SessionControls(props: SessionControlsProps) {
  const { sessions, sessionId, runId, runActive, phase, onSession, onRun, onRefresh, onError } = props;
  const [files, setFiles] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const session = sessions.find((s) => s.id === sessionId) ?? null;
  const run = session?.runs.find((r) => r.id === runId) ?? null;

  useEffect(() => {
    api.audioFiles().then(setFiles).catch(() => setFiles([]));
  }, [sessions.length]);

  async function act(task: () => Promise<unknown>) {
    setBusy(true);
    onError(null);
    try {
      await task();
      await onRefresh();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const start = (source: Source, speed: number | null = null) =>
    act(async () => onRun((await api.startRun({ session_id: sessionId!, source, speed })).id));

  return (
    <div className="controls" data-testid="controls">
      <select
        value={sessionId ?? ""}
        onChange={(e) => onSession(e.target.value ? Number(e.target.value) : null)}
        aria-label="Session"
      >
        <option value="">Pick a session</option>
        {sessions.map((s) => (
          <option key={s.id} value={s.id}>
            {s.name} ({s.recorded_at})
          </option>
        ))}
      </select>
      {session && (
        <select value={runId ?? ""} onChange={(e) => onRun(e.target.value ? Number(e.target.value) : null)} aria-label="Run">
          <option value="">No run</option>
          {session.runs.map((r) => (
            <option key={r.id} value={r.id}>
              Run {r.id}: {r.kind}, {r.status}
              {r.accepted ? ", accepted" : ""}
            </option>
          ))}
        </select>
      )}

      {session && !runActive && (
        <>
          <button disabled={busy} onClick={() => start("mic")}>
            Start live
          </button>
          {session.audio_uri && (
            <>
              <button disabled={busy} onClick={() => start("file", 1)}>
                Replay 1×
              </button>
              <button disabled={busy} onClick={() => start("file", null)}>
                Replay fast
              </button>
            </>
          )}
          <button className="secondary" disabled={busy} onClick={() => start("text")}>
            Typed run
          </button>
          {run && run.kind !== "merge" && !run.accepted && run.status !== "running" && (
            <button className="secondary" disabled={busy} onClick={() => act(() => api.accept(run.id))}>
              Accept run
            </button>
          )}
        </>
      )}
      {runActive && runId !== null && (
        <>
          {phase === "intro" && (
            <button disabled={busy} onClick={() => act(() => api.introDone(runId))}>
              Done with introductions
            </button>
          )}
          <button className="danger" disabled={busy} onClick={() => act(() => api.stop(runId))}>
            Stop
          </button>
        </>
      )}

      <details className="new-session">
        <summary>New session</summary>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!name.trim()) return;
            void act(async () => {
              const created = await api.newSession({ name: name.trim() });
              setName("");
              onSession(created.id);
            });
          }}
        >
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Session name" />
          <button disabled={busy}>Create</button>
        </form>
        <label className="upload">
          Upload a recording
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void act(async () => onSession((await api.upload(file)).id));
            }}
          />
        </label>
        {files.length > 0 && (
          <div className="samples">
            From a file:
            {files.map((file) => (
              <button
                key={file}
                className="link"
                onClick={() =>
                  act(async () => onSession((await api.newSession({ name: file.split("/").pop()!, audio_uri: file })).id))
                }
              >
                {file.split("/").pop()}
              </button>
            ))}
          </div>
        )}
      </details>
    </div>
  );
}
