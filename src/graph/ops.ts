import type { PersonId, PersonAttributes, RelationshipCategory, Gender } from "./types";

// The LLM's output contract. Each op is one mutation the model proposes; the
// reducer applies them deterministically. People are addressed by id;
// relationships by their (source, target) pair — the reducer canonicalizes.
export type GraphOp =
  | {
      op: "add_person";
      id: PersonId;
      name: string;
      aliases?: string[];
      gender?: Gender | null;
      attributes?: PersonAttributes | null;
    }
  | {
      op: "update_person";
      id: PersonId;
      name?: string | null;
      addAliases?: string[];
      gender?: Gender | null;
      attributes?: PersonAttributes | null;
    }
  | { op: "remove_person"; id: PersonId }
  | {
      op: "add_relationship";
      source: PersonId;
      target: PersonId;
      category: RelationshipCategory;
      label: string; // short, 2-3 words
      description?: string | null; // full detail
      directed?: boolean;
      strength?: number | null;
    }
  | {
      // category identifies which edge on the pair to update (not changed here)
      op: "update_relationship";
      source: PersonId;
      target: PersonId;
      category: RelationshipCategory;
      label?: string | null;
      description?: string | null;
      strength?: number | null;
    }
  | {
      op: "remove_relationship";
      source: PersonId;
      target: PersonId;
      category: RelationshipCategory;
    }
  | {
      // Change a relationship's category atomically (e.g. acquaintance -> romantic).
      op: "retype_relationship";
      source: PersonId;
      target: PersonId;
      fromCategory: RelationshipCategory; // the existing category
      toCategory: RelationshipCategory; // the new category
      label?: string | null; // optional new short label
      description?: string | null;
      strength?: number | null;
    };
