import type { SocialGraph } from "./types";

// Compact representation of the current graph for the LLM prompt. Giving the
// model the existing people (with aliases) is what lets it reuse ids instead of
// creating duplicate nodes for "Sarah" vs "Sarah Chen".
export function serializeGraphForPrompt(graph: SocialGraph): string {
  const people = Object.values(graph.people).map((p) => ({
    id: p.id,
    name: p.name,
    aliases: p.aliases,
    ...(p.gender ? { gender: p.gender } : {}),
    ...(p.attributes ? { attributes: p.attributes } : {}),
  }));

  const relationships = Object.values(graph.relationships).map((r) => ({
    source: r.source,
    target: r.target,
    category: r.category,
    label: r.label,
    ...(r.description ? { description: r.description } : {}),
    directed: r.directed,
    ...(r.strength != null ? { strength: r.strength } : {}),
  }));

  return JSON.stringify({ people, relationships }, null, 2);
}
