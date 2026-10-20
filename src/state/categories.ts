import type { Category, FactKind, Gender } from "./types";

export interface CategoryMeta {
  color: string;
  weight: number;
}

// Edge color and the share of edge weight that the category alone gives.
export const CATEGORIES: Record<Category, CategoryMeta> = {
  romantic: { color: "#f472b6", weight: 1 },
  family: { color: "#a78bfa", weight: 0.9 },
  friend: { color: "#34d399", weight: 0.7 },
  professional: { color: "#60a5fa", weight: 0.4 },
  other: { color: "#94a3b8", weight: 0.3 },
};

export const CATEGORY_LIST = Object.keys(CATEGORIES) as Category[];

export const KIND_COLORS: Record<FactKind, string> = {
  personal: "#60a5fa",
  relationship: "#34d399",
  story: "#fbbf24",
  membership: "#a78bfa",
  correction: "#94a3b8",
};

// Node fill, as a light and a dark stop of a gradient.
export const GENDER_COLORS: Record<Gender, [string, string]> = {
  male: ["#7dd3fc", "#2563eb"],
  female: ["#f9a8d4", "#db2777"],
  unknown: ["#cbd5e1", "#475569"],
};

export const GROUP_COLORS = ["#a78bfa", "#34d399", "#fbbf24", "#60a5fa", "#f472b6", "#2dd4bf"];

export function initials(name: string): string {
  const words = name.split(" ").filter(Boolean);
  return (words[0][0] + (words.length > 1 ? words[words.length - 1][0] : "")).toUpperCase();
}
