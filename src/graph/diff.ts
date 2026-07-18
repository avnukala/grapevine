import type { Person, Relationship, SocialGraph } from "./types";

// The actual changes between two graph states — the ground truth of what a
// request did, including relationship removals cascaded by remove_person.
export interface GraphDiff {
  peopleAdded: Person[];
  peopleRemoved: Person[];
  peopleUpdated: { before: Person; after: Person }[];
  relsAdded: Relationship[];
  relsRemoved: Relationship[];
  relsUpdated: { before: Relationship; after: Relationship }[];
}

function personChanged(a: Person, b: Person): boolean {
  return (
    a.name !== b.name ||
    a.aliases.join("") !== b.aliases.join("") ||
    JSON.stringify(a.attributes ?? {}) !== JSON.stringify(b.attributes ?? {})
  );
}

function relChanged(a: Relationship, b: Relationship): boolean {
  return (
    a.category !== b.category ||
    a.label !== b.label ||
    (a.description ?? "") !== (b.description ?? "") ||
    a.directed !== b.directed ||
    (a.strength ?? null) !== (b.strength ?? null)
  );
}

export function diffGraphs(before: SocialGraph, after: SocialGraph): GraphDiff {
  const diff: GraphDiff = {
    peopleAdded: [],
    peopleRemoved: [],
    peopleUpdated: [],
    relsAdded: [],
    relsRemoved: [],
    relsUpdated: [],
  };

  for (const [id, p] of Object.entries(after.people)) {
    const prev = before.people[id];
    if (!prev) diff.peopleAdded.push(p);
    else if (personChanged(prev, p)) diff.peopleUpdated.push({ before: prev, after: p });
  }
  for (const [id, p] of Object.entries(before.people)) {
    if (!after.people[id]) diff.peopleRemoved.push(p);
  }

  for (const [id, r] of Object.entries(after.relationships)) {
    const prev = before.relationships[id];
    if (!prev) diff.relsAdded.push(r);
    else if (relChanged(prev, r)) diff.relsUpdated.push({ before: prev, after: r });
  }
  for (const [id, r] of Object.entries(before.relationships)) {
    if (!after.relationships[id]) diff.relsRemoved.push(r);
  }

  return diff;
}

export function diffCount(d: GraphDiff): number {
  return (
    d.peopleAdded.length +
    d.peopleRemoved.length +
    d.peopleUpdated.length +
    d.relsAdded.length +
    d.relsRemoved.length +
    d.relsUpdated.length
  );
}
