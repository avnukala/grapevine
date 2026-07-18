import type { Person, Relationship, RelationshipCategory, SocialGraph } from "./types";
import type { GraphOp } from "./ops";
import { defaultDirected } from "./relationshipTypes";

// Identity is (unordered pair + category), so two people can hold several
// relationships at once (e.g. professional "business partner" AND other
// "roommate"). Re-describing the same pair+category updates that edge — so a
// correction within a category ("dating" -> "married", both romantic) is not a
// duplicate.
export function relationshipId(a: string, b: string, category: RelationshipCategory): string {
  return [a, b].sort().join("__") + "::" + category;
}

function slugify(s: string): string {
  return s
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function clone(graph: SocialGraph): SocialGraph {
  return {
    people: { ...graph.people },
    relationships: { ...graph.relationships },
  };
}

// Defensive: if the model references a person id that doesn't exist yet (e.g. it
// added an edge before the node), materialize a minimal person rather than dropping the op.
function ensurePerson(graph: SocialGraph, id: string): void {
  if (!graph.people[id]) {
    graph.people[id] = {
      id,
      name: id
        .split("-")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" "),
      aliases: [],
    };
  }
}

function mergeAliases(existing: string[], incoming: string[] | undefined): string[] {
  if (!incoming || incoming.length === 0) return existing;
  const seen = new Set(existing.map((a) => a.toLowerCase()));
  const merged = [...existing];
  for (const a of incoming) {
    const t = a.trim();
    if (t && !seen.has(t.toLowerCase())) {
      seen.add(t.toLowerCase());
      merged.push(t);
    }
  }
  return merged;
}

function applyOne(graph: SocialGraph, op: GraphOp): void {
  switch (op.op) {
    case "add_person": {
      const id = op.id || slugify(op.name);
      const existing = graph.people[id];
      if (existing) {
        // Idempotent: merge instead of duplicating.
        graph.people[id] = {
          ...existing,
          name: op.name || existing.name,
          aliases: mergeAliases(existing.aliases, op.aliases),
          attributes: { ...existing.attributes, ...(op.attributes ?? {}) },
        };
      } else {
        const person: Person = {
          id,
          name: op.name,
          aliases: op.aliases ?? [],
        };
        if (op.attributes) person.attributes = op.attributes;
        graph.people[id] = person;
      }
      break;
    }

    case "update_person": {
      const existing = graph.people[op.id];
      if (!existing) {
        ensurePerson(graph, op.id);
      }
      const person = graph.people[op.id];
      graph.people[op.id] = {
        ...person,
        name: op.name ?? person.name,
        aliases: mergeAliases(person.aliases, op.addAliases),
        attributes: { ...person.attributes, ...(op.attributes ?? {}) },
      };
      break;
    }

    case "remove_person": {
      delete graph.people[op.id];
      // Clean up dangling relationships.
      for (const [rid, rel] of Object.entries(graph.relationships)) {
        if (rel.source === op.id || rel.target === op.id) {
          delete graph.relationships[rid];
        }
      }
      break;
    }

    case "add_relationship": {
      if (op.source === op.target) break; // no self-loops
      ensurePerson(graph, op.source);
      ensurePerson(graph, op.target);
      const id = relationshipId(op.source, op.target, op.category);
      const existing = graph.relationships[id];
      const rel: Relationship = {
        id,
        source: op.source,
        target: op.target,
        category: op.category,
        label: op.label,
        directed: op.directed ?? defaultDirected(op.category),
        ...(op.description != null ? { description: op.description } : {}),
        ...(op.strength != null ? { strength: op.strength } : {}),
      };
      // Merge onto the existing edge if there is one (idempotent + corrections).
      graph.relationships[id] = existing ? { ...existing, ...rel } : rel;
      break;
    }

    case "update_relationship": {
      // category identifies which edge on the pair to update; it is not changed
      // here (to change the kind, remove the old edge and add the new one).
      const id = relationshipId(op.source, op.target, op.category);
      const existing = graph.relationships[id];
      if (!existing) break; // nothing to update
      graph.relationships[id] = {
        ...existing,
        label: op.label ?? existing.label,
        description: op.description ?? existing.description,
        strength: op.strength ?? existing.strength,
      };
      break;
    }

    case "remove_relationship": {
      delete graph.relationships[relationshipId(op.source, op.target, op.category)];
      break;
    }
  }
}

// Pure: returns a new graph with the ops applied. Never mutates the input.
export function applyOps(graph: SocialGraph, ops: GraphOp[]): SocialGraph {
  const next = clone(graph);
  for (const op of ops) {
    try {
      applyOne(next, op);
    } catch {
      // A malformed op shouldn't take down the whole batch.
    }
  }
  return next;
}
