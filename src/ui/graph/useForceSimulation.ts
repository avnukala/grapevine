import { useEffect, useRef, type MutableRefObject } from "react";
import { forceSimulation, forceLink, forceManyBody, forceCollide, forceX, forceY, type Simulation } from "d3-force";
import type { XYPosition } from "@xyflow/react";
import type { SocialGraph } from "../../graph/types";
import { estimateNodeSize } from "./geometry";

export interface SimNode {
  id: string;
  x: number;
  y: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  r: number;
}

interface SimLink {
  source: string;
  target: string;
}

const avg = (ns: number[]) => ns.reduce((a, b) => a + b, 0) / ns.length;

export function useForceSimulation(opts: {
  onTick: (positions: Map<string, XYPosition>) => void;
  draggingRef: MutableRefObject<string | null>;
}) {
  const onTickRef = useRef(opts.onTick);
  onTickRef.current = opts.onTick;
  const draggingRef = opts.draggingRef;

  const simRef = useRef<Simulation<SimNode, SimLink> | null>(null);
  const rafRef = useRef<number | null>(null);
  const runIdRef = useRef(0);
  const positionsRef = useRef<Map<string, XYPosition>>(new Map());

  // Create the simulation once; it never self-runs (`.stop()`), driven only by
  // the manual rAF loop below so freeze-at-rest and single-`setNodes`-per-frame
  // batching both fall out for free.
  useEffect(() => {
    const sim = forceSimulation<SimNode>([])
      .force("link", forceLink<SimNode, SimLink>([]).id((d) => d.id).distance(130).strength(0.35))
      .force("charge", forceManyBody().strength(-1400).distanceMax(700))
      .force("collide", forceCollide<SimNode>().radius((d) => d.r + 10).iterations(2))
      .force("x", forceX(0).strength(0.05))
      .force("y", forceY(0).strength(0.05))
      .stop();
    simRef.current = sim;

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      sim.stop();
      simRef.current = null;
    };
  }, []);

  function readPositions(): Map<string, XYPosition> {
    const sim = simRef.current;
    if (!sim) return positionsRef.current;
    const map = positionsRef.current;
    for (const n of sim.nodes()) {
      map.set(n.id, { x: n.x, y: n.y });
    }
    return map;
  }

  function ensureLoop() {
    if (rafRef.current !== null) return;
    const runId = runIdRef.current;
    const frame = () => {
      const sim = simRef.current;
      if (!sim) return;
      sim.tick();
      onTickRef.current(readPositions());
      if (sim.alpha() < sim.alphaMin() && draggingRef.current === null) {
        rafRef.current = null;
        if (runIdRef.current === runId) {
          for (const n of sim.nodes()) {
            if (n.id !== draggingRef.current) {
              n.fx = null;
              n.fy = null;
            }
          }
        }
      } else {
        rafRef.current = requestAnimationFrame(frame);
      }
    };
    rafRef.current = requestAnimationFrame(frame);
  }

  // Reconcile the sim's node/link set with the current graph. `addedIds` are
  // newly-added people this pass — the established graph is locked (fx/fy) so
  // only newcomers settle; centroid-seeded so they ease in from a sensible spot.
  function sync(graph: SocialGraph, addedIds: Set<string>, center: XYPosition) {
    const sim = simRef.current;
    if (!sim) return;
    runIdRef.current += 1;

    const existing = new Map(sim.nodes().map((n) => [n.id, n]));
    const people = Object.values(graph.people);
    const nextNodes: SimNode[] = people.map((p) => {
      const prev = existing.get(p.id);
      const size = estimateNodeSize(p);
      const r = Math.max(size.width, size.height) / 2;
      if (prev) {
        prev.r = r;
        return prev;
      }
      const seed = seedPosition(p.id, graph, positionsRef.current, center);
      return { id: p.id, x: seed.x, y: seed.y, r };
    });

    // forceLink mutates link.source/target from ids into node object references,
    // so map to fresh literals rather than passing Relationship objects — those
    // are the actual PersonId strings living in graph.relationships.
    const nextLinks: SimLink[] = Object.values(graph.relationships).map((r) => ({
      source: r.source,
      target: r.target,
    }));

    sim.nodes(nextNodes);
    (sim.force("link") as ReturnType<typeof forceLink<SimNode, SimLink>>).links(nextLinks);
    sim.force("x", forceX<SimNode>(center.x).strength(0.05));
    sim.force("y", forceY<SimNode>(center.y).strength(0.05));

    if (addedIds.size > 0) {
      for (const n of nextNodes) {
        if (!addedIds.has(n.id)) {
          n.fx = n.x;
          n.fy = n.y;
        } else {
          n.fx = null;
          n.fy = null;
        }
      }
      sim.alpha(0.8);
    }

    ensureLoop();
  }

  function pin(id: string, pos: XYPosition) {
    const sim = simRef.current;
    if (!sim) return;
    draggingRef.current = id;
    const node = sim.nodes().find((n) => n.id === id);
    if (node) {
      node.fx = pos.x;
      node.fy = pos.y;
      node.x = pos.x;
      node.y = pos.y;
    }
    sim.alphaTarget(0.3);
    ensureLoop();
  }

  function release(id: string) {
    const sim = simRef.current;
    if (!sim) return;
    if (draggingRef.current === id) draggingRef.current = null;
    const node = sim.nodes().find((n) => n.id === id);
    if (node) {
      node.fx = null;
      node.fy = null;
    }
    sim.alphaTarget(0);
    ensureLoop();
  }

  return { sync, pin, release, positionsRef, draggingRef };
}

// Seed a new node at the centroid of its already-placed neighbors so it eases in
// from a sensible spot instead of flying in from the origin; fall back to the
// viewport/graph center when it has no placed neighbor yet.
function seedPosition(
  id: string,
  graph: SocialGraph,
  positions: Map<string, XYPosition>,
  center: XYPosition,
): XYPosition {
  const jitter = () => (Math.random() - 0.5) * 60;
  const placed = Object.values(graph.relationships)
    .filter((r) => r.source === id || r.target === id)
    .map((r) => positions.get(r.source === id ? r.target : r.source))
    .filter((p): p is XYPosition => p !== undefined);
  if (placed.length > 0) {
    return { x: avg(placed.map((p) => p.x)) + jitter(), y: avg(placed.map((p) => p.y)) + jitter() };
  }
  return { x: center.x + jitter(), y: center.y + jitter() };
}
