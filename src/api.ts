import type { Change, ModelCall, Person, Prompt, Run, Segment, Session } from "./state/types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? undefined : { "content-type": "application/json" },
  });
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail ?? JSON.stringify(body)).catch(() => "");
    throw new Error(`${response.status} ${detail || response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function post<T>(path: string, body: unknown = {}): Promise<T> {
  return call<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export type Source = "file" | "mic" | "text";

export const api = {
  sessions: () => call<Session[]>("/api/sessions"),
  audioFiles: () => call<string[]>("/api/audio-files"),
  newSession: (body: { name: string; audio_uri?: string | null; intro_end_ms?: number | null; roster?: string[] }) =>
    post<Session>("/api/sessions", body),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return call<Session>("/api/sessions/upload", { method: "POST", body: form });
  },
  updateSession: (id: number, body: { intro_end_ms?: number | null; name?: string }) =>
    call<Session>(`/api/sessions/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  startRun: (body: { session_id: number; source: Source; speed?: number | null; config?: Record<string, unknown> }) =>
    post<Run>("/api/runs", body),
  activeRun: () => call<Run | null>("/api/runs/active"),
  run: (id: number) => call<Run & { session: Session | null }>(`/api/runs/${id}`),
  changes: (id: number) => call<Change[]>(`/api/runs/${id}/changes`),
  modelCalls: (id: number) => call<ModelCall[]>(`/api/runs/${id}/model_calls`),
  stop: (id: number) => post<Run>(`/api/runs/${id}/stop`),
  introDone: (id: number) => post<Run>(`/api/runs/${id}/intro-done`),
  accept: (id: number) => post<Run>(`/api/runs/${id}/accept`),
  text: (id: number, speaker: string, text: string) => post<Segment>(`/api/runs/${id}/text`, { speaker, text }),
  answer: (
    runId: number,
    promptId: number,
    body: { person_id?: number; new_name?: string; nickname?: string; keep_fact_id?: number | null },
  ) => post<Prompt>(`/api/runs/${runId}/prompts/${promptId}/answer`, body),
  dismiss: (runId: number, promptId: number) => post<Prompt>(`/api/runs/${runId}/prompts/${promptId}/dismiss`),
  assignSpeaker: (
    runId: number,
    segmentId: number,
    body: { person_id?: number; new_name?: string; every_line?: boolean },
  ) => post<Segment>(`/api/runs/${runId}/segments/${segmentId}/speaker`, body),
  editLine: (runId: number, segmentId: number, text: string) =>
    call<Segment>(`/api/runs/${runId}/segments/${segmentId}`, { method: "PATCH", body: JSON.stringify({ text }) }),
  removePerson: (runId: number, personId: number) =>
    call<null>(`/api/runs/${runId}/people/${personId}`, { method: "DELETE" }),
  people: () => call<Person[]>("/api/people"),
  merge: () => post<Run>("/api/merge"),
  global: () => call<{ run: Run | null }>("/api/global"),
  resolveMerge: (runId: number, promptId: number, body: { action: string; fact_id?: number }) =>
    post<Prompt>(`/api/merge/${runId}/prompts/${promptId}`, body),
};

export function audioUrl(sessionId: number): string {
  return `/api/sessions/${sessionId}/audio`;
}
