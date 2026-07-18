import {
  RELATIONSHIP_CATEGORIES,
  CATEGORY_LIST,
} from "../src/graph/relationshipTypes.ts";

// ---- System prompt -------------------------------------------------------

const categoryGuide = CATEGORY_LIST.map((c) => {
  const meta = RELATIONSHIP_CATEGORIES[c];
  return `- "${c}": ${meta.description} Examples: ${meta.examples.join(", ")}.`;
}).join("\n");

export const SYSTEM_PROMPT = `You maintain a live social graph from a person thinking out loud about people they know.

NODES are people. EDGES are relationships between two people.

You receive (1) the CURRENT GRAPH as JSON and (2) NEW TEXT the user just said. Return a list of OPERATIONS that update the graph to reflect the new text. Only emit operations for information in the new text — do not restate the whole graph.

IDENTITY / DEDUPLICATION — this is the most important rule:
- Before adding a person, check the CURRENT GRAPH's people and their aliases. If the individual already exists (by name, nickname, or pronoun referring to them), REUSE their existing "id". Do NOT create a new node.
- When the user refers to an existing person by a new name or nickname, use "update_person" with "addAliases" instead of creating a duplicate.
- Person ids are stable kebab-case slugs of the person's full name, e.g. "sarah-chen". Once assigned, never change an id.

RELATIONSHIPS:
- Every relationship has a "category" (for grouping/color) and a specific "label" (the exact wording). Categories:
${categoryGuide}
- "label" is a natural, plain-language summary of HOW the two people relate — the word you'd use to describe the relationship out loud. 1-3 words, a noun or short noun phrase. Examples: "brother", "coworker", "ex-girlfriend", "love interest", "small fling", "childhood friend", "boss", "roommate", "college roommate". It is NOT a truncation of the description and NOT a full sentence — write a clean relationship term, even if the source text is long or vague.
- "description" holds the FULL detail — a complete sentence with any nuance, history, or context from the text (e.g. "Dated for three years in college, broke up but stayed close friends"). This is shown when the user clicks a person, not on the edge.
- Put the best-fitting bucket in "category".
- Set "directed": true only when direction matters (e.g. "reports to", "mentors"). Symmetric relationships ("friends", "married") are undirected.
- Two people can have MULTIPLE relationships at once, as long as they are different categories — e.g. business partners AND roommates is a "professional" edge plus an "other" edge. Add one relationship per distinct kind.
- Identity is (pair of people + category). There is at most one relationship per pair per category. So "update_relationship" and "remove_relationship" take the "category" to say WHICH edge on the pair they mean.
- A correction WITHIN a category updates that edge (e.g. "dating" -> "married" are both "romantic" — update it, don't add a second romantic edge). To change a relationship into a different category, remove the old edge (by its category) and add the new one.

WEIGHT / STRENGTH:
- Always assign every relationship a "strength" from 1 to 5 (integer): 1 = weak or distant, 3 = ordinary, 5 = very strong or close.
- Infer it from the language: "best friend", "married", "inseparable", "childhood friend" → 4-5; "friend", "colleague" → 3; "acquaintance", "met once", "barely knows" → 1-2.
- When the user later signals the bond changed ("they've grown close", "they drifted apart"), use "update_relationship" to adjust the existing strength — don't add a new edge.

CORRECTIONS:
- "no wait, connect A to C not B" => remove the wrong relationship and add the right one.
- Handle removals and updates, not just additions.

If the new text contains no graph-relevant information, return an empty "operations" array.`;

// ---- Structured-output schema -------------------------------------------

// Structured outputs require every listed property to be in "required" and
// additionalProperties:false. Optional fields are expressed as nullable types.
const attributesSchema = {
  type: ["object", "null"],
  additionalProperties: false,
  required: ["role", "org", "notes"],
  properties: {
    role: { type: ["string", "null"] },
    org: { type: ["string", "null"] },
    notes: { type: ["string", "null"] },
  },
};

const categoryEnum = { type: "string", enum: CATEGORY_LIST };

const opVariants = [
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "id", "name", "aliases", "attributes"],
    properties: {
      op: { type: "string", const: "add_person" },
      id: { type: "string", description: "kebab-case slug of the person's name" },
      name: { type: "string" },
      aliases: { type: "array", items: { type: "string" } },
      attributes: attributesSchema,
    },
  },
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "id", "name", "addAliases", "attributes"],
    properties: {
      op: { type: "string", const: "update_person" },
      id: { type: "string" },
      name: { type: ["string", "null"] },
      addAliases: { type: "array", items: { type: "string" } },
      attributes: attributesSchema,
    },
  },
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "id"],
    properties: {
      op: { type: "string", const: "remove_person" },
      id: { type: "string" },
    },
  },
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "source", "target", "category", "label", "description", "directed", "strength"],
    properties: {
      op: { type: "string", const: "add_relationship" },
      source: { type: "string", description: "existing or new person id" },
      target: { type: "string", description: "existing or new person id" },
      category: categoryEnum,
      label: {
        type: "string",
        description:
          "natural 1-3 word relationship term, e.g. 'brother', 'coworker', 'ex-girlfriend', 'love interest', 'small fling'",
      },
      description: {
        type: ["string", "null"],
        description: "full detail sentence shown in the popup; null if nothing beyond the label",
      },
      directed: { type: "boolean" },
      strength: {
        type: "integer",
        description: "relationship weight: 1 (weak) to 5 (strong). Always provide a value.",
      },
    },
  },
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "source", "target", "category", "label", "description", "strength"],
    properties: {
      op: { type: "string", const: "update_relationship" },
      source: { type: "string" },
      target: { type: "string" },
      category: { ...categoryEnum, description: "identifies which relationship on the pair to update" },
      label: { type: ["string", "null"], description: "short 2-3 word caption, or null to keep" },
      description: { type: ["string", "null"], description: "full detail, or null to keep" },
      strength: { type: ["integer", "null"] },
    },
  },
  {
    type: "object",
    additionalProperties: false,
    required: ["op", "source", "target", "category"],
    properties: {
      op: { type: "string", const: "remove_relationship" },
      source: { type: "string" },
      target: { type: "string" },
      category: { ...categoryEnum, description: "identifies which relationship on the pair to remove" },
    },
  },
];

export const OPS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["operations"],
  properties: {
    operations: {
      type: "array",
      items: { anyOf: opVariants },
    },
  },
};

export function buildUserMessage(graphJson: string, newText: string): string {
  return `CURRENT GRAPH:
${graphJson}

NEW TEXT:
"""
${newText}
"""

Return the operations that update the graph to reflect the NEW TEXT.`;
}
