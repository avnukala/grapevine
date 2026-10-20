import type { Category } from "./types";

export interface CategoryMeta {
  color: string;
  weight: number;
}

// Edge color and the share of edge weight that the category alone gives.
export const CATEGORIES: Record<Category, CategoryMeta> = {
  romantic: { color: "#e64980", weight: 1 },
  family: { color: "#7048e8", weight: 0.9 },
  friend: { color: "#2f9e44", weight: 0.7 },
  professional: { color: "#1c7ed6", weight: 0.4 },
  other: { color: "#868e96", weight: 0.3 },
};

export const CATEGORY_LIST = Object.keys(CATEGORIES) as Category[];
