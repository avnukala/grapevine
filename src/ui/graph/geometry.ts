import type { InternalNode } from "@xyflow/react";
import type { Person } from "../../graph/types";

// Fixed pill height (see PersonNode.tsx); width is measured per person.
export const NODE_HEIGHT = 36;
const BADGE_WIDTH = 24;
const GAP = 8; // between badge and name
const H_PADDING = 12; // left + right padding inside the pill

// Must match PersonNode's name text CSS exactly (13px/600).
const NAME_FONT = "600 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

let measureCtx: CanvasRenderingContext2D | null = null;
function getMeasureCtx(): CanvasRenderingContext2D {
  if (!measureCtx) {
    measureCtx = document.createElement("canvas").getContext("2d");
  }
  return measureCtx!;
}

function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  const first = parts[0][0] ?? "";
  const last = parts.length > 1 ? (parts[parts.length - 1][0] ?? "") : "";
  return (first + last).toUpperCase();
}

const sizeCache = new Map<string, { width: number; height: number }>();

// Deterministic pill size so the force sim's collide radius agrees with what
// actually renders — node.measured is populated async, too late for tick 1.
export function estimateNodeSize(person: Person): { width: number; height: number } {
  const key = `${person.id}:${person.name}`;
  const existing = sizeCache.get(key);
  if (existing) return existing;

  const ctx = getMeasureCtx();
  ctx.font = NAME_FONT;
  const nameWidth = ctx.measureText(person.name).width;
  const width = Math.ceil(H_PADDING * 2 + BADGE_WIDTH + GAP + nameWidth);
  const size = { width, height: NODE_HEIGHT };
  sizeCache.set(key, size);
  return size;
}

export function personInitials(person: Person): string {
  return initials(person.name);
}

// ---- Floating edges: intersection of the line between two node centres with
// each node's own rectangle border. ----

interface NodeLike {
  internals: { positionAbsolute: { x: number; y: number } };
  measured: { width?: number; height?: number };
}

function nodeCenter(node: NodeLike): { x: number; y: number } {
  const w = node.measured.width ?? 0;
  const h = node.measured.height ?? 0;
  return {
    x: node.internals.positionAbsolute.x + w / 2,
    y: node.internals.positionAbsolute.y + h / 2,
  };
}

// Intersection of the segment from `from`'s centre to `to`'s centre with
// `from`'s own rectangle border. Each node uses its own half-extents for its
// own centre — the official React Flow example bug reuses one node's
// half-extents for both, which visibly misattaches edges when pill widths differ.
function rectIntersection(from: NodeLike, to: NodeLike): { x: number; y: number } {
  const w = (from.measured.width ?? 0) / 2;
  const h = (from.measured.height ?? 0) / 2;
  const c1 = nodeCenter(from);
  const c2 = nodeCenter(to);

  const dx = c2.x - c1.x;
  const dy = c2.y - c1.y;
  if (dx === 0 && dy === 0) return c1;

  // Scale so the point lands exactly on the rectangle border.
  const scale = 1 / Math.max(Math.abs(dx) / w, Math.abs(dy) / h);
  return { x: c1.x + dx * scale, y: c1.y + dy * scale };
}

export function getEdgeParams(
  source: InternalNode,
  target: InternalNode,
): { sx: number; sy: number; tx: number; ty: number } {
  const sourceIntersection = rectIntersection(source, target);
  const targetIntersection = rectIntersection(target, source);
  return {
    sx: sourceIntersection.x,
    sy: sourceIntersection.y,
    tx: targetIntersection.x,
    ty: targetIntersection.y,
  };
}

// Perpendicular offset for the Nth of M parallel edges between the same pair,
// applied at the midpoint so a single edge stays straight and parallels fan out.
export function parallelOffset(
  sx: number,
  sy: number,
  tx: number,
  ty: number,
  parallelIndex: number,
  parallelCount: number,
): { mx: number; my: number } {
  const mx = (sx + tx) / 2;
  const my = (sy + ty) / 2;
  if (parallelCount <= 1) return { mx, my };

  const dx = tx - sx;
  const dy = ty - sy;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const offset = (parallelIndex - (parallelCount - 1) / 2) * 26;
  return { mx: mx + nx * offset, my: my + ny * offset };
}
