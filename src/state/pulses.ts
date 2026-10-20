import type { Pulse } from "../ui/GraphView";
import type { Change, RunState } from "./types";

export const PULSE_MS = 5000;

export interface Recent {
  pulses: Pulse[];
  factIds: number[];
  eventIds: number[];
}

// What changed in the last few seconds. `at` gives each change its time on the same clock as `now`:
// wall time for a live run, audio time for a replay on the debug screen.
export function recent(changes: Change[], at: (change: Change) => number, now: number): Recent {
  const pulses: Pulse[] = [];
  const factIds: number[] = [];
  const eventIds: number[] = [];
  for (const change of changes) {
    const age = now - at(change);
    if (age < 0 || age >= PULSE_MS) continue;
    const op = change.op;
    if (op.type === "fact") {
      factIds.push(op.fact.id);
      pulses.push({
        key: `f${change.seq}`,
        personIds: op.fact.people.filter((p) => p.role !== "teller").map((p) => p.person_id),
        relationshipIds: [],
        text: op.fact.text,
      });
    }
    if (op.type === "relationship") {
      pulses.push({
        key: `r${change.seq}`,
        personIds: [op.relationship.a, op.relationship.b],
        relationshipIds: [op.relationship.id],
        text: null,
      });
    }
    if (op.type === "event") eventIds.push(op.event.id);
  }
  return { pulses, factIds, eventIds };
}

export function speakerAt(state: RunState, ms: number): number | null {
  const segment = Object.values(state.segments).find((s) => s.start_ms <= ms && ms < s.end_ms);
  return segment?.person_id ?? null;
}
