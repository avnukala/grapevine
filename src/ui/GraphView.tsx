import * as d3 from "d3";
import { useEffect, useMemo, useRef, useState } from "react";
import { CATEGORIES } from "../state/categories";
import { canonicalId, edgeWeight, nodeLabels, roomIds, visiblePeople } from "../state/derive";
import type { WeightFactors } from "../state/derive";
import type { Gender, Relationship, RunState } from "../state/types";

export interface Pulse {
  key: string;
  personIds: number[];
  relationshipIds: number[];
  text: string | null;
}

interface GraphViewProps {
  state: RunState;
  pulses: Pulse[];
  speakingId: number | null;
  factors: WeightFactors;
  selectedId: number | null;
  onSelect: (id: number | null) => void;
}

interface Node extends d3.SimulationNodeDatum {
  id: number;
  mass: number;
  inRoom: boolean;
  pvx: number;
  pvy: number;
}

interface Edge {
  rel: Relationship;
  source: Node;
  target: Node;
  weight: number;
  bend: number;
}

interface PairLink extends d3.SimulationLinkDatum<Node> {
  key: string;
  weight: number;
}

const GENDER_COLORS: Record<Gender, string> = { male: "#4c8bf5", female: "#f06595", unknown: "#adb5bd" };
const ROOM_MASS = 12;
const OUTSIDE_MASS = 1;
const BEND_PX = 26;

function pairKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

// A heavy node takes 1/mass of each push, so people in the room drift to their place and never jump.
function inertia(nodes: Node[], phase: "start" | "end"): d3.Force<Node, undefined> {
  return () => {
    for (const node of nodes) {
      if (phase === "start") {
        node.pvx = node.vx ?? 0;
        node.pvy = node.vy ?? 0;
      } else {
        node.vx = node.pvx + ((node.vx ?? 0) - node.pvx) / node.mass;
        node.vy = node.pvy + ((node.vy ?? 0) - node.pvy) / node.mass;
      }
    }
  };
}

function curve(edge: Edge): string {
  const { source: s, target: t, bend } = edge;
  const [sx, sy, tx, ty] = [s.x ?? 0, s.y ?? 0, t.x ?? 0, t.y ?? 0];
  const length = Math.hypot(tx - sx, ty - sy) || 1;
  const cx = (sx + tx) / 2 - ((ty - sy) / length) * bend;
  const cy = (sy + ty) / 2 + ((tx - sx) / length) * bend;
  return `M${sx},${sy} Q${cx},${cy} ${tx},${ty}`;
}

function midpoint(edge: Edge): [number, number] {
  const { source: s, target: t, bend } = edge;
  const [sx, sy, tx, ty] = [s.x ?? 0, s.y ?? 0, t.x ?? 0, t.y ?? 0];
  const length = Math.hypot(tx - sx, ty - sy) || 1;
  return [(sx + tx) / 2 - ((ty - sy) / length) * (bend / 2), (sy + ty) / 2 + ((tx - sx) / length) * (bend / 2)];
}

export function GraphView({ state, pulses, speakingId, factors, selectedId, onSelect }: GraphViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const nodesRef = useRef(new Map<number, Node>());
  const simRef = useRef<d3.Simulation<Node, PairLink> | null>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const svg = svgRef.current!;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(svg);
    const root = d3.select(svg);
    const layer = root.select<SVGGElement>("g.layer");
    root.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on("zoom", (event) => layer.attr("transform", event.transform)),
    );
    return () => {
      observer.disconnect();
      simRef.current?.stop();
    };
  }, []);

  const derived = {
    people: visiblePeople(state),
    room: roomIds(state).map((id) => canonicalId(state, id)),
    labels: nodeLabels(state),
    relationships: Object.values(state.relationships).map((r) => ({
      ...r,
      a: canonicalId(state, r.a),
      b: canonicalId(state, r.b),
    })),
  };
  // The simulation restarts only when what it draws changes, not on every change op.
  const structure = JSON.stringify([
    derived.people.map((p) => [p.id, p.gender]),
    derived.room,
    derived.labels,
    derived.relationships,
  ]);
  const { people, room: roomList, labels, relationships } = useMemo(() => derived, [structure]);
  const room = useMemo(() => new Set(roomList), [roomList]);
  const genders = useMemo(() => Object.fromEntries(people.map((p) => [p.id, p.gender])), [people]);

  useEffect(() => {
    const { width, height } = size;
    const known = nodesRef.current;
    const roomOrder = [...room];
    const nodes: Node[] = people.map((person) => {
      const existing = known.get(person.id);
      const inRoom = room.has(person.id);
      if (existing) {
        existing.inRoom = inRoom;
        existing.mass = inRoom ? ROOM_MASS : OUTSIDE_MASS;
        return existing;
      }
      const index = roomOrder.indexOf(person.id);
      const spread = Math.min(width, height);
      const angle = inRoom
        ? (2 * Math.PI * index) / Math.max(roomOrder.length, 1) - Math.PI / 2
        : ((person.id * 137.5) % 360) * (Math.PI / 180);
      const radius = inRoom ? spread * 0.2 : spread * 0.42;
      const node: Node = {
        id: person.id,
        mass: inRoom ? ROOM_MASS : OUTSIDE_MASS,
        inRoom,
        pvx: 0,
        pvy: 0,
        x: width / 2 + radius * Math.cos(angle),
        y: height / 2 + radius * Math.sin(angle),
      };
      known.set(person.id, node);
      return node;
    });
    const byId = new Map(nodes.map((n) => [n.id, n]));

    const perPair = new Map<string, Relationship[]>();
    for (const rel of relationships) {
      if (!byId.has(rel.a) || !byId.has(rel.b) || rel.a === rel.b) continue;
      const key = pairKey(rel.a, rel.b);
      perPair.set(key, [...(perPair.get(key) ?? []), rel]);
    }
    const edges: Edge[] = [];
    const links: PairLink[] = [];
    for (const [key, rels] of perPair) {
      rels.forEach((rel, i) =>
        edges.push({
          rel,
          source: byId.get(rel.a)!,
          target: byId.get(rel.b)!,
          weight: edgeWeight(rel, factors),
          bend: (i - (rels.length - 1) / 2) * BEND_PX,
        }),
      );
      const weight = Math.max(...rels.map((r) => edgeWeight(r, factors)));
      links.push({ key, source: byId.get(rels[0].a)!, target: byId.get(rels[0].b)!, weight });
    }

    const svg = d3.select(svgRef.current!);
    const edgePaths = svg
      .select("g.edges")
      .selectAll<SVGPathElement, Edge>("path")
      .data(edges, (e) => String(e.rel.id))
      .join("path")
      .attr("data-id", (e) => e.rel.id)
      .attr("stroke", (e) => CATEGORIES[e.rel.category].color)
      .attr("stroke-width", (e) => 1.5 + e.weight * 4)
      .attr("stroke-dasharray", (e) => (e.rel.status === "past" ? "6 5" : null));
    edgePaths.selectAll("title").data((e) => [e]).join("title").text((e) => `${e.rel.label}: ${e.rel.summary ?? ""}`);
    const edgeLabels = svg
      .select("g.edge-labels")
      .selectAll<SVGTextElement, Edge>("text")
      .data(edges, (e) => String(e.rel.id))
      .join("text")
      .attr("fill", (e) => CATEGORIES[e.rel.category].color)
      .text((e) => e.rel.label);

    const nodeGroups = svg
      .select("g.nodes")
      .selectAll<SVGGElement, Node>("g.node")
      .data(nodes, (n) => String(n.id))
      .join((enter) => {
        const g = enter.append("g").attr("class", "node");
        g.append("circle");
        g.append("text").attr("class", "node-label").attr("text-anchor", "middle");
        return g;
      })
      .attr("data-id", (n) => n.id)
      .classed("in-room", (n) => n.inRoom)
      .on("click", (event, n) => {
        event.stopPropagation();
        onSelect(n.id);
      });
    nodeGroups
      .select("circle")
      .attr("r", (n) => (n.inRoom ? 24 : 18))
      .attr("fill", (n) => GENDER_COLORS[genders[n.id]]);
    nodeGroups
      .select("text")
      .attr("dy", (n) => (n.inRoom ? 40 : 33))
      .text((n) => labels[n.id]);
    nodeGroups.call(
      d3
        .drag<SVGGElement, Node>()
        .on("start", (event, n) => {
          if (!event.active) simRef.current?.alphaTarget(0.2).restart();
          n.fx = n.x;
          n.fy = n.y;
        })
        .on("drag", (event, n) => {
          n.fx = event.x;
          n.fy = event.y;
        })
        .on("end", (event, n) => {
          if (!event.active) simRef.current?.alphaTarget(0);
          n.fx = null;
          n.fy = null;
        }),
    );
    svg.on("click", () => onSelect(null));

    const ticked = () => {
      edgePaths.attr("d", curve);
      edgeLabels.attr("x", (e) => midpoint(e)[0]).attr("y", (e) => midpoint(e)[1] - 4);
      nodeGroups.attr("transform", (n) => `translate(${n.x},${n.y})`);
      svg
        .select("g.captions")
        .selectAll<SVGTextElement, Edge>("text")
        .attr("x", (e) => midpoint(e)[0])
        .attr("y", (e) => midpoint(e)[1] + 14);
    };

    simRef.current?.stop();
    simRef.current = d3
      .forceSimulation<Node, PairLink>(nodes)
      .force("inertia-start", inertia(nodes, "start"))
      .force(
        "link",
        d3
          .forceLink<Node, PairLink>(links)
          .id((n) => String(n.id))
          .distance((l) => 70 + (1 - l.weight) * 200)
          .strength((l) => 0.15 + l.weight * 0.6),
      )
      .force("charge", d3.forceManyBody<Node>().strength(-320))
      .force("x", d3.forceX<Node>(width / 2).strength(0.03))
      .force("y", d3.forceY<Node>(height / 2).strength(0.03))
      .force("collide", d3.forceCollide<Node>(40))
      .force("inertia-end", inertia(nodes, "end"))
      .alpha(0.6)
      .alphaDecay(0.015)
      .on("tick", ticked);
    ticked();
  }, [people, room, labels, relationships, genders, factors, size, onSelect]);

  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    const litPeople = new Set(pulses.flatMap((p) => p.personIds.map((id) => canonicalId(state, id))));
    const litPairs = new Map<string, string | null>();
    const litRelationships = new Set(pulses.flatMap((p) => p.relationshipIds));
    for (const pulse of pulses) {
      const ids = pulse.personIds.map((id) => canonicalId(state, id));
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          const key = pairKey(ids[i], ids[j]);
          if (pulse.text || !litPairs.has(key)) litPairs.set(key, pulse.text);
        }
      }
    }
    const edges = svg.select("g.edges").selectAll<SVGPathElement, Edge>("path");
    edges.classed(
      "lit",
      (e) => litRelationships.has(e.rel.id) || litPairs.has(pairKey(e.source.id, e.target.id)),
    );
    const captioned = new Map<string, Edge>();
    edges.each((e) => {
      const key = pairKey(e.source.id, e.target.id);
      if (litPairs.get(key) && !captioned.has(key)) captioned.set(key, e);
    });
    svg
      .select("g.captions")
      .selectAll<SVGTextElement, Edge>("text")
      .data([...captioned.values()], (e) => String(e.rel.id))
      .join("text")
      .attr("x", (e) => midpoint(e)[0])
      .attr("y", (e) => midpoint(e)[1] + 14)
      .text((e) => shorten(litPairs.get(pairKey(e.source.id, e.target.id)) ?? ""));
    svg
      .select("g.nodes")
      .selectAll<SVGGElement, Node>("g.node")
      .classed("speaking", (n) => speakingId !== null && n.id === canonicalId(state, speakingId))
      .classed("mentioned", (n) => litPeople.has(n.id))
      .classed("selected", (n) => n.id === selectedId);
  }, [pulses, speakingId, selectedId, state, people, relationships, size]);

  return (
    <svg ref={svgRef} className="graph" data-testid="graph">
      <g className="layer">
        <g className="edges" />
        <g className="edge-labels" />
        <g className="captions" />
        <g className="nodes" />
      </g>
    </svg>
  );
}

function shorten(text: string, max = 48): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
