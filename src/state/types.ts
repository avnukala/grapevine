// The records of a run, as the backend writes them in each change op.

export type Gender = "male" | "female" | "unknown";
export type Category = "romantic" | "family" | "friend" | "professional" | "other";
export type FactKind = "personal" | "relationship" | "story" | "membership" | "correction";
export type DatePrecision = "year" | "season" | "month" | "day";
export type Phase = "intro" | "conversation" | "review" | "ended";

export interface Person {
  id: number;
  name: string;
  aliases: string[];
  gender: Gender;
  merged_into_id: number | null;
}

export interface Segment {
  id: number;
  start_ms: number;
  end_ms: number;
  speaker_label: string;
  person_id: number | null;
  speaker_confidence: number | null;
  speaker_manual: boolean;
  text: string;
  timings: Partial<Record<"detect_ms" | "transcribe_ms" | "embed_ms" | "extract_ms" | "total_ms", number>>;
}

export interface PersonRole {
  person_id: number;
  role: string;
}

export interface Fact {
  id: number;
  kind: FactKind;
  text: string;
  detail: string | null;
  told_by_person_id: number | null;
  when_text: string | null;
  start_date: string | null;
  end_date: string | null;
  date_precision: DatePrecision | null;
  event_id: number | null;
  confidence: number;
  disputed: boolean;
  revised_by_fact_id: number | null;
  people: PersonRole[];
  segment_ids: number[];
}

export interface GraphEvent {
  id: number;
  name: string;
  when_text: string | null;
  start_date: string | null;
  end_date: string | null;
  date_precision: DatePrecision | null;
  place: string | null;
  people: PersonRole[];
}

export interface Group {
  id: number;
  name: string;
  aliases: string[];
  description: string | null;
}

export interface Member {
  group_id: number;
  person_id: number;
  since: string | null;
  until: string | null;
  source_fact_id: number | null;
}

export interface Relationship {
  id: number;
  a: number;
  b: number;
  category: Category;
  label: string;
  summary: string | null;
  status: "current" | "past";
  met_on: string | null;
  met_event_id: number | null;
  closeness: number | null;
  recency: number | null;
  sentiment: number | null;
  evidence: number[];
}

export interface Prompt {
  id: number;
  text: string;
  reason: string;
  person_ids: number[];
  shown_at_ms: number;
  dismissed_at_ms: number | null;
  payload: Record<string, unknown>;
}

export type Op =
  | { type: "person"; person: Person }
  | { type: "speaker"; speaker: { label: string; person_id: number | null } }
  | { type: "segment"; segment: Segment }
  | { type: "fact"; fact: Fact }
  | { type: "event"; event: GraphEvent }
  | { type: "event_removed"; id: number }
  | { type: "group"; group: Group }
  | { type: "member"; member: Member }
  | { type: "member_removed"; group_id: number; person_id: number }
  | { type: "relationship"; relationship: Relationship }
  | { type: "relationship_removed"; id: number }
  | { type: "person_removed"; id: number }
  | { type: "prompt"; prompt: Prompt }
  | { type: "phase"; phase: Phase }
  | { type: "roster"; person_ids: number[] };

export interface Change {
  seq: number;
  audio_ms: number;
  segment_id: number | null;
  model_call_id: number | null;
  op: Op;
}

export interface RunState {
  people: Record<number, Person>;
  speakers: Record<string, number | null>;
  segments: Record<number, Segment>;
  facts: Record<number, Fact>;
  events: Record<number, GraphEvent>;
  groups: Record<number, Group>;
  members: Record<string, Member>;
  relationships: Record<number, Relationship>;
  prompts: Record<number, Prompt>;
  roster: number[];
  phase: Phase;
}

export interface Run {
  id: number;
  session_id: number | null;
  kind: "live" | "replay" | "merge";
  accepted: number;
  status: "running" | "done" | "stopped" | "failed";
  created_at: string;
  code_version: string;
  active?: boolean;
  config?: Record<string, unknown>;
}

export interface Session {
  id: number;
  name: string;
  recorded_at: string;
  audio_uri: string | null;
  notes: string | null;
  intro_end_ms: number | null;
  roster: string[];
  runs: Run[];
}

export interface ModelCall {
  id: number;
  worker: string;
  model: string;
  model_file_hash: string;
  prompt_version: string;
  seed: number;
  input_hash: string;
  latency_ms: number;
  audio_ms: number | null;
  cached: boolean;
  request: { messages: { role: string; content: string }[] };
  response: string;
}
