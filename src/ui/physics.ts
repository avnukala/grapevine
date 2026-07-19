// Unified force simulation for the social graph.
//
// A single continuous d3-force simulation owns every node position — the initial
// settle, incremental additions, weight changes, and live dragging all share one
// equilibrium, so the graph never jumps between competing layout engines.
//
// The core requirement is that edge strength is SIGNED. Positive affinity pulls
// a pair together (stronger = closer and stiffer); negative affinity actively
// pushes them apart to a keep-away distance that grows with hostility. Stock
// force layouts (cose, cola) treat every edge as attractive, which is why this
// module exists: bonds use d3's link springs, feuds get a custom repulsive force.

import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
} from "d3-force";
import type { Simulation, SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import type { Core } from "cytoscape";
import type { SocialGraph } from "../graph/types";

export interface SimNode extends SimulationNodeDatum {
  id: string;
  r: number; // collision radius; tracks the rendered node size
}

// A pair of people can hold several relationships at once (e.g. professional +
// roommate). The physics acts on one link per unordered pair, weighted by the
// mean signed strength of everything between them.
interface PairLink extends SimulationLinkDatum<SimNode> {
  source: SimNode;
  target: SimNode;
  weight: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

// ---- Signed-weight mappings ----

// Rest length for a positive/neutral bond. +5 hugs (~65px between centers),
// a neutral 0 is a long loose tie (~230px). The simulation honors these
// exactly — no cap like cose's ~320px ideal-length ceiling.
function bondDistance(w: number): number {
  return 230 - 33 * clamp(w, 0, 5);
}

// Keep-away radius for a negative bond: hostile pairs are pushed until they sit
// at least this far apart. -1 → 325px, -5 → 585px — visibly farther than any
// friendly tie, so antagonism reads at a glance.
function feudDistance(w: number): number {
  return 260 + 65 * clamp(-w, 0, 5);
}

// Node display radius grows gently with how many people someone is tied to, so
// social hubs read as hubs. Shared with the renderer so collision matches pixels.
export function nodeRadius(degree: number): number {
  return clamp(11 + 3.2 * Math.sqrt(degree), 13, 24);
}

// The force d3 doesn't ship: signed-edge repulsion. While a hostile pair is
// inside its feud radius they are pushed apart (harder the more hostile); past
// the radius a faint tether pulls back so the still-real relationship edge stays
// readable on screen instead of stretching indefinitely.
function forceFeuds(feuds: PairLink[]) {
  return (alpha: number) => {
    for (const { source: a, target: b, weight } of feuds) {
      const R = feudDistance(weight);
      let dx = b.x! - a.x!;
      let dy = b.y! - a.y!;
      const d = Math.hypot(dx, dy) || 1;
      // Same spring form as d3's forceLink; push-apart gain dwarfs the tether
      // so "apart" is always the visual outcome.
      const gain = d < R ? 0.07 + 0.03 * clamp(-weight, 0, 5) : 0.015;
      const k = ((R - d) / d) * gain * alpha;
      dx *= k;
      dy *= k;
      b.vx! += dx;
      b.vy! += dy;
      a.vx! -= dx;
      a.vy! -= dy;
    }
  };
}

export class GraphPhysics {
  private cy: Core;
  private sim: Simulation<SimNode, PairLink>;
  private nodes: SimNode[] = [];
  private byId = new Map<string, SimNode>();

  constructor(cy: Core) {
    this.cy = cy;
    this.sim = forceSimulation<SimNode>([])
      // Ambient personal space; distanceMax so far-apart clusters stop shoving
      // each other and the feud radius stays the dominant long-range force.
      .force("charge", forceManyBody<SimNode>().strength(-360).distanceMax(560))
      // Weak pull toward the origin keeps disconnected components and repelled
      // rivals from drifting out of frame.
      .force("center-x", forceX(0).strength(0.035))
      .force("center-y", forceY(0).strength(0.035))
      // Hard floor: nodes plus a margin for their name labels never overlap.
      .force("collide", forceCollide<SimNode>((n) => n.r + 16).strength(0.8))
      .alpha(0);
    this.sim.stop();
    this.sim.on("tick", () => this.write());
  }

  // Reconcile the simulation with the graph: existing nodes keep position and
  // velocity, newcomers are seeded (or left to d3's spiral placement), departed
  // nodes drop out, and the per-pair signed forces are rebuilt.
  sync(graph: SocialGraph, seed?: (id: string) => { x: number; y: number }) {
    const pairs = new Map<string, { a: string; b: string; sum: number; n: number }>();
    for (const r of Object.values(graph.relationships)) {
      const [a, b] = [r.source, r.target].sort();
      const key = `${a}|${b}`;
      const p = pairs.get(key) ?? { a, b, sum: 0, n: 0 };
      p.sum += r.strength ?? 3;
      p.n += 1;
      pairs.set(key, p);
    }
    // degree = number of distinct people someone is tied to (multi-edges count once)
    const degree = new Map<string, number>();
    for (const { a, b } of pairs.values()) {
      degree.set(a, (degree.get(a) ?? 0) + 1);
      degree.set(b, (degree.get(b) ?? 0) + 1);
    }

    const ids = new Set(Object.keys(graph.people));
    this.nodes = this.nodes.filter((n) => ids.has(n.id));
    this.byId = new Map(this.nodes.map((n) => [n.id, n]));
    for (const id of ids) {
      let n = this.byId.get(id);
      if (!n) {
        n = { id, r: nodeRadius(0) };
        const p = seed?.(id);
        if (p) {
          n.x = p.x;
          n.y = p.y;
        }
        this.nodes.push(n);
        this.byId.set(id, n);
      }
      n.r = nodeRadius(degree.get(id) ?? 0);
    }
    this.sim.nodes(this.nodes);

    const bonds: PairLink[] = [];
    const feuds: PairLink[] = [];
    for (const { a, b, sum, n } of pairs.values()) {
      const sa = this.byId.get(a);
      const sb = this.byId.get(b);
      if (!sa || !sb) continue;
      const weight = sum / n;
      (weight < 0 ? feuds : bonds).push({ source: sa, target: sb, weight });
    }
    const minDeg = (l: PairLink) =>
      Math.max(1, Math.min(degree.get(l.source.id) ?? 1, degree.get(l.target.id) ?? 1));
    this.sim.force(
      "bonds",
      forceLink<SimNode, PairLink>(bonds)
        .distance((l) => bondDistance(l.weight))
        // Stiffer springs for stronger bonds, divided by the smaller endpoint's
        // degree so a popular hub doesn't crush its whole circle inward.
        .strength((l) => Math.min(1, (0.45 + 0.11 * Math.max(0, l.weight)) / minDeg(l))),
    );
    this.sim.force("feuds", feuds.length ? forceFeuds(feuds) : null);
  }

  // Run the simulation to rest synchronously — used for the very first fill so
  // the graph appears already settled instead of assembling on screen. Manual
  // tick() doesn't fire the tick event, so positions are written once at the end.
  settleNow() {
    this.sim.alpha(1);
    for (let i = 0; i < 300 && this.sim.alpha() > this.sim.alphaMin(); i++) this.sim.tick();
    this.sim.alpha(0);
    this.write();
  }

  // Re-energize after a structural change. Nodes already at equilibrium barely
  // move; anything newly added or re-weighted glides to its place.
  reheat(alpha = 0.55) {
    this.sim.alpha(Math.max(alpha, this.sim.alpha())).alphaTarget(0).restart();
  }

  // Drag protocol (mirrors the d3 convention): grab pins the node where it is
  // but does NOT wake the sim — a plain click must not jiggle the graph. The
  // first real drag movement raises alphaTarget so neighbors respond live, and
  // release lets everything cool back to a freeze.
  grab(id: string) {
    const n = this.byId.get(id);
    if (!n) return;
    n.fx = n.x;
    n.fy = n.y;
  }

  drag(id: string, x: number, y: number) {
    const n = this.byId.get(id);
    if (!n) return;
    n.fx = x;
    n.fy = y;
    if (this.sim.alphaTarget() < 0.25) this.sim.alphaTarget(0.28).restart();
  }

  free(id: string) {
    const n = this.byId.get(id);
    if (n) {
      n.fx = null;
      n.fy = null;
    }
    this.sim.alphaTarget(0); // finish the current settle, then freeze
  }

  destroy() {
    this.sim.stop();
  }

  private write() {
    const cy = this.cy;
    if (cy.destroyed()) return;
    cy.batch(() => {
      for (const n of this.nodes) {
        const el = cy.getElementById(n.id);
        if (el.nonempty()) el.position({ x: n.x!, y: n.y! });
      }
    });
  }
}
