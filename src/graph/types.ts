// The source-of-truth data model for the social graph.
// Nodes are people; edges are relationships between two people.

export type PersonId = string; // stable kebab-case slug, e.g. "sarah-chen"

export interface PersonAttributes {
  role?: string; // "manager", "student", "barista"
  org?: string; // "Acme Corp", "NYU"
  notes?: string; // anything else worth remembering
}

export type Gender = "male" | "female" | "unknown";

export interface Person {
  id: PersonId;
  name: string; // display name, e.g. "Sarah Chen"
  aliases: string[]; // ["Sarah", "she", "Ms. Chen"] — the main lever for dedup
  gender?: Gender; // inferred from name / pronouns / relationship terms; drives node color
  attributes?: PersonAttributes;
}

// Categories drive edge styling and give the model a small, fixed vocabulary.
// The specific relationship (e.g. "childhood friend", "girlfriend") lives in
// Relationship.label — category groups it for color/legend purposes.
export type RelationshipCategory =
  | "romantic"
  | "family"
  | "friend"
  | "acquaintance"
  | "professional"
  | "other";

export interface Relationship {
  id: string; // canonical, derived from the two person ids (see reducer)
  source: PersonId;
  target: PersonId;
  category: RelationshipCategory;
  label: string; // SHORT edge label, 2-3 words: "girlfriend", "childhood friend", "reports to"
  description?: string; // full detail, shown in the node popup (not on the edge)
  directed: boolean; // true => arrow from source -> target (e.g. "mentors")
  strength?: number; // signed affinity -5..5: + pulls nodes together, - repels them
}

export interface SocialGraph {
  people: Record<PersonId, Person>;
  relationships: Record<string, Relationship>;
}

export function emptyGraph(): SocialGraph {
  return { people: {}, relationships: {} };
}
