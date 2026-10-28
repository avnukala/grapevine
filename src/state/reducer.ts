import type { Change, Op, RunState } from "./types";

// Mirrors `Graph.apply` in backend/grapevine/graph.py: each op carries the full record it writes.

export function emptyState(): RunState {
  return {
    people: {},
    speakers: {},
    segments: {},
    facts: {},
    events: {},
    groups: {},
    members: {},
    relationships: {},
    prompts: {},
    roster: [],
    phase: "intro",
  };
}

export function applyOp(state: RunState, op: Op): RunState {
  switch (op.type) {
    case "person":
      return { ...state, people: { ...state.people, [op.person.id]: op.person } };
    case "speaker":
      return { ...state, speakers: { ...state.speakers, [op.speaker.label]: op.speaker.person_id } };
    case "segment":
      return { ...state, segments: { ...state.segments, [op.segment.id]: op.segment } };
    case "fact":
      return { ...state, facts: { ...state.facts, [op.fact.id]: op.fact } };
    case "event":
      return { ...state, events: { ...state.events, [op.event.id]: op.event } };
    case "event_removed": {
      const events = { ...state.events };
      delete events[op.id];
      return { ...state, events };
    }
    case "group":
      return { ...state, groups: { ...state.groups, [op.group.id]: op.group } };
    case "member": {
      const key = `${op.member.group_id}:${op.member.person_id}:${op.member.source_fact_id}`;
      return { ...state, members: { ...state.members, [key]: op.member } };
    }
    case "member_removed":
      return removeMember(state, op.group_id, op.person_id);
    case "relationship":
      return { ...state, relationships: { ...state.relationships, [op.relationship.id]: op.relationship } };
    case "relationship_removed": {
      const relationships = { ...state.relationships };
      delete relationships[op.id];
      return { ...state, relationships };
    }
    case "person_removed":
      return removePerson(state, op.id);
    case "prompt":
      return { ...state, prompts: { ...state.prompts, [op.prompt.id]: op.prompt } };
    case "phase":
      return { ...state, phase: op.phase };
    case "roster":
      return { ...state, roster: op.person_ids };
  }
}

// A group lives in a run through its members, so it leaves with the last one.
function removeMember(state: RunState, groupId: number, personId: number): RunState {
  const members = Object.fromEntries(
    Object.entries(state.members).filter(([, m]) => m.group_id !== groupId || m.person_id !== personId),
  );
  if (Object.values(members).some((m) => m.group_id === groupId)) return { ...state, members };
  const groups = { ...state.groups };
  delete groups[groupId];
  return { ...state, members, groups };
}

function removePerson(state: RunState, id: number): RunState {
  const people = { ...state.people };
  delete people[id];
  const names = (roles: { person_id: number }[]) => roles.some((p) => p.person_id === id);
  const events = Object.fromEntries(
    Object.entries(state.events).map(([key, e]) => [key, { ...e, people: e.people.filter((p) => p.person_id !== id) }]),
  );
  return {
    ...state,
    people,
    roster: state.roster.filter((p) => p !== id),
    members: Object.fromEntries(Object.entries(state.members).filter(([, m]) => m.person_id !== id)),
    facts: Object.fromEntries(Object.entries(state.facts).filter(([, f]) => !names(f.people))),
    events,
  };
}

// Changes arrive in write order, which is not audio order: extraction lags transcription.
export function stateAt(changes: Change[], untilMs = Infinity): RunState {
  let state = emptyState();
  for (const change of changes) {
    if (change.audio_ms <= untilMs) state = applyOp(state, change.op);
  }
  return state;
}
