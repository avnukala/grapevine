import type { RelationshipCategory } from "./types";

export interface CategoryMeta {
  category: RelationshipCategory;
  color: string; // edge color in the graph + legend swatch
  defaultDirected: boolean; // typical directedness when the model is unsure
  description: string; // shown to the LLM so it picks the right category
  examples: string[]; // concrete labels that fall under this category
}

// The single source of truth for edge categories. Used by:
//  - the LLM prompt (enumerate categories + examples so it classifies well)
//  - the reducer (default directedness)
//  - the renderer + legend (color)
export const RELATIONSHIP_CATEGORIES: Record<RelationshipCategory, CategoryMeta> = {
  romantic: {
    category: "romantic",
    color: "#e64980",
    defaultDirected: false,
    description: "Romantic or intimate partnerships, current or past.",
    examples: ["boyfriend", "girlfriend", "married to", "engaged", "dating", "ex-partner"],
  },
  family: {
    category: "family",
    color: "#7048e8",
    defaultDirected: false,
    description: "Blood relatives or family by marriage.",
    examples: ["mother", "father", "sibling", "cousin", "aunt", "grandparent"],
  },
  friend: {
    category: "friend",
    color: "#2f9e44",
    defaultDirected: false,
    description: "Friendships of any closeness.",
    examples: ["friend", "childhood friend", "best friend", "close friend"],
  },
  acquaintance: {
    category: "acquaintance",
    color: "#f08c00",
    defaultDirected: false,
    description: "People who know each other loosely.",
    examples: ["acquaintance", "neighbor", "met once", "friend of a friend"],
  },
  professional: {
    category: "professional",
    color: "#1c7ed6",
    defaultDirected: false,
    description: "Work or school connections. Often directed (e.g. reporting lines).",
    examples: ["colleague", "coworker", "manager", "reports to", "mentors", "classmate", "client"],
  },
  other: {
    category: "other",
    color: "#868e96",
    defaultDirected: false,
    description: "Any connection that does not fit the categories above.",
    examples: ["knows", "roommate", "rival", "landlord"],
  },
};

export const CATEGORY_LIST = Object.keys(RELATIONSHIP_CATEGORIES) as RelationshipCategory[];

export function categoryColor(category: RelationshipCategory): string {
  return RELATIONSHIP_CATEGORIES[category]?.color ?? RELATIONSHIP_CATEGORIES.other.color;
}

export function defaultDirected(category: RelationshipCategory): boolean {
  return RELATIONSHIP_CATEGORIES[category]?.defaultDirected ?? false;
}
