import type { Category, FactKind, Gender } from "./types";

export interface CategoryMeta {
  color: string;
  weight: number;
}

// Edge color and the share of edge weight that the category alone gives.
export const CATEGORIES: Record<Category, CategoryMeta> = {
  romantic: { color: "#c4746e", weight: 1 },
  family: { color: "#a292a3", weight: 0.9 },
  friend: { color: "#87a987", weight: 0.7 },
  professional: { color: "#8ba4b0", weight: 0.4 },
  other: { color: "#7a8382", weight: 0.3 },
};

export const CATEGORY_LIST = Object.keys(CATEGORIES) as Category[];

export const KIND_COLORS: Record<FactKind, string> = {
  personal: "#8ba4b0",
  relationship: "#87a987",
  story: "#c4b28a",
  membership: "#a292a3",
  correction: "#7a8382",
};

// Colors from Kanagawa Dragon. They are mid-tones, so text on them uses the dark ink color.
export const GENDER_COLORS: Record<Gender, string> = {
  male: "#8ba4b0",
  female: "#a292a3",
  unknown: "#9e9b93",
};

export const GROUP_COLORS = ["#a292a3", "#87a987", "#c4b28a", "#8ba4b0", "#c4746e", "#8ea4a2"];

export function initials(name: string): string {
  const words = name.split(" ").filter(Boolean);
  return (words[0][0] + (words.length > 1 ? words[words.length - 1][0] : "")).toUpperCase();
}
