import * as d3 from "d3";
import { useEffect, useMemo, useRef, useState } from "react";
import { CATEGORIES, GENDER_COLORS, GROUP_COLORS, initials } from "../state/categories";
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

interface Hull {
  id: number;
  name: string;
  members: number[];
  color: string;
}

const ROOM_MASS = 12;
const OUTSIDE_MASS = 1;
const BEND_PX = 28;
const ROOM_RADIUS = 26;
const OUTSIDE_RADIUS = 19;
const HULL_PAD = 46;
const CHAR_PX = 6.4;
const GROUP_PULL = 0.08;

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

// Pulls the members of a named group toward their center, so groups form clusters.
function cohesion(hulls: Hull[], byId: Map<number, Node>): d3.Force<Node, undefined> {
  return (alpha) => {
    for (const hull of hulls) {
      const members = hull.members.map((id) => byId.get(id)!);
      const cx = d3.mean(members, (n) => n.x) ?? 0;
      const cy = d3.mean(members, (n) => n.y) ?? 0;
      for (const node of members) {
        node.vx = (node.vx ?? 0) + (cx - (node.x ?? 0)) * GROUP_PULL * alpha;
        node.vy = (node.vy ?? 0) + (cy - (node.y ?? 0)) * GROUP_PULL * alpha;
      }
    }
  };
}

function control(edge: Edge, share: number): [number, number] {
  const { source: s, target: t, bend } = edge;
  const [sx, sy, tx, ty] = [s.x ?? 0, s.y ?? 0, t.x ?? 0, t.y ?? 0];
  const length = Math.hypot(tx - sx, ty - sy) || 1;
  return [(sx + tx) / 2 - ((ty - sy) / length) * bend * share, (sy + ty) / 2 + ((tx - sx) / length) * bend * share];
}

function curve(edge: Edge): string {
  const [cx, cy] = control(edge, 1);
  return `M${edge.source.x},${edge.source.y} Q${cx},${cy} ${edge.target.x},${edge.target.y}`;
}

function hullPath(points: [number, number][]): string {
  const padded = points.flatMap(([x, y]) =>
    d3.range(8).map((i): [number, number] => [
      x + HULL_PAD * Math.cos((i * Math.PI) / 4),
      y + HULL_PAD * Math.sin((i * Math.PI) / 4),
    ]),
  );
  const hull = d3.polygonHull(padded) ?? padded;
  return d3.line().curve(d3.curveCatmullRomClosed.alpha(0.8))(hull) ?? "";
}

function radius(node: Node): number {
  return node.inRoom ? ROOM_RADIUS : OUTSIDE_RADIUS;
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
    hulls: Object.values(state.groups)
      .map((group, i): Hull => ({
        id: group.id,
        name: group.name,
        color: GROUP_COLORS[i % GROUP_COLORS.length],
        members: [
          ...new Set(
            Object.values(state.members)
              .filter((m) => m.group_id === group.id)
              .map((m) => canonicalId(state, m.person_id)),
          ),
        ],
      }))
      .filter((hull) => hull.members.length >= 2),
  };
  // The simulation restarts only when what it draws changes, not on every change op.
  const structure = JSON.stringify([
    derived.people.map((p) => [p.id, p.gender, p.name]),
    derived.room,
    derived.labels,
    derived.relationships,
    derived.hulls,
  ]);
  const { people, room: roomList, labels, relationships, hulls } = useMemo(() => derived, [structure]);
  const room = useMemo(() => new Set(roomList), [roomList]);
  const genders = useMemo(() => Object.fromEntries(people.map((p) => [p.id, p.gender])), [people]);
  const names = useMemo(() => Object.fromEntries(people.map((p) => [p.id, p.name])), [people]);

  useEffect(() => {
    const { width, height } = size;
    const known = nodesRef.current;
    const nodes: Node[] = people.map((person) => {
      const existing = known.get(person.id);
      const inRoom = room.has(person.id);
      if (existing) {
        existing.inRoom = inRoom;
        existing.mass = inRoom ? ROOM_MASS : OUTSIDE_MASS;
        return existing;
      }
      const index = roomList.indexOf(person.id);
      const spread = Math.min(width, height);
      const angle = inRoom
        ? (2 * Math.PI * index) / Math.max(roomList.length, 1) - Math.PI / 2
        : ((person.id * 137.5) % 360) * (Math.PI / 180);
      const distance = inRoom ? spread * 0.2 : spread * 0.42;
      const node: Node = {
        id: person.id,
        mass: inRoom ? ROOM_MASS : OUTSIDE_MASS,
        inRoom,
        pvx: 0,
        pvy: 0,
        x: width / 2 + distance * Math.cos(angle),
        y: height / 2 + distance * Math.sin(angle),
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
    const groupHulls = hulls.filter((h) => h.members.every((id) => byId.has(id)));

    const svg = d3.select(svgRef.current!);
    const hullPaths = svg
      .select("g.hulls")
      .selectAll<SVGGElement, Hull>("g.hull")
      .data(groupHulls, (h) => String(h.id))
      .join((enter) => {
        const g = enter.append("g").attr("class", "hull");
        g.append("path");
        g.append("text");
        return g;
      });
    hullPaths.select("path").attr("fill", (h) => h.color).attr("stroke", (h) => h.color);
    hullPaths.select("text").attr("fill", (h) => h.color).text((h) => h.name);

    const edgePaths = svg
      .select("g.edges")
      .selectAll<SVGPathElement, Edge>("path")
      .data(edges, (e) => String(e.rel.id))
      .join("path")
      .attr("data-id", (e) => e.rel.id)
      .attr("stroke", (e) => CATEGORIES[e.rel.category].color)
      .attr("stroke-width", (e) => 1.2 + e.weight * 3.5)
      .attr("stroke-dasharray", (e) => (e.rel.status === "past" ? "5 6" : null));
    edgePaths
      .selectAll("title")
      .data((e) => [e])
      .join("title")
      .text((e) => `${e.rel.label}${e.rel.summary ? `: ${e.rel.summary}` : ""}`);
    const edgeLabels = svg
      .select("g.edge-labels")
      .selectAll<SVGTextElement, Edge>("text")
      .data(edges, (e) => String(e.rel.id))
      .join("text")
      .attr("fill", (e) => CATEGORIES[e.rel.category].color)
      .text((e) => e.rel.label.toLowerCase());

    const nodeGroups = svg
      .select("g.nodes")
      .selectAll<SVGGElement, Node>("g.node")
      .data(nodes, (n) => String(n.id))
      .join((enter) => {
        const g = enter.append("g").attr("class", "node").style("opacity", 0);
        g.transition().duration(600).style("opacity", 1);
        g.append("circle").attr("class", "ripple");
        g.append("circle").attr("class", "ripple late");
        g.append("circle").attr("class", "halo");
        g.append("circle").attr("class", "body");
        g.append("text").attr("class", "initials").attr("text-anchor", "middle").attr("dy", "0.35em");
        const label = g.append("g").attr("class", "label");
        label.append("rect").attr("rx", 9);
        label.append("text").attr("text-anchor", "middle").attr("dy", "0.35em");
        return g;
      })
      .attr("data-id", (n) => n.id)
      .classed("in-room", (n) => n.inRoom)
      .on("click", (event, n) => {
        event.stopPropagation();
        onSelect(n.id);
      });
    nodeGroups.selectAll<SVGCircleElement, Node>("circle.ripple, circle.halo").attr("r", (n) => radius(n) + 3);
    nodeGroups
      .select("circle.body")
      .attr("r", radius)
      .attr("fill", (n) => `url(#node-${genders[n.id] as Gender})`);
    nodeGroups
      .select("text.initials")
      .attr("font-size", (n) => (n.inRoom ? 14 : 11))
      .text((n) => initials(names[n.id]));
    nodeGroups.select("g.label").attr("transform", (n) => `translate(0,${radius(n) + 15})`);
    nodeGroups
      .select("g.label rect")
      .attr("x", (n) => -(labels[n.id].length * CHAR_PX + 16) / 2)
      .attr("y", -9)
      .attr("width", (n) => labels[n.id].length * CHAR_PX + 16)
      .attr("height", 18);
    nodeGroups.select("g.label text").text((n) => labels[n.id]);
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
      hullPaths
        .select("path")
        .attr("d", (h) => hullPath(h.members.map((id) => [byId.get(id)!.x ?? 0, byId.get(id)!.y ?? 0])));
      hullPaths
        .select("text")
        .attr("x", (h) => d3.mean(h.members, (id) => byId.get(id)!.x) ?? 0)
        .attr("y", (h) => (d3.min(h.members, (id) => byId.get(id)!.y) ?? 0) - HULL_PAD + 14);
      edgePaths.attr("d", curve);
      edgeLabels.attr("x", (e) => control(e, 0.5)[0]).attr("y", (e) => control(e, 0.5)[1] - 5);
      nodeGroups.attr("transform", (n) => `translate(${n.x},${n.y})`);
      svg
        .select("g.captions")
        .selectAll<SVGGElement, Edge>("g")
        .attr("transform", (e) => `translate(${control(e, 0.5)[0]},${control(e, 0.5)[1] + 16})`);
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
          .distance((l) => 80 + (1 - l.weight) * 200)
          .strength((l) => 0.15 + l.weight * 0.6),
      )
      .force("charge", d3.forceManyBody<Node>().strength(-380))
      .force("x", d3.forceX<Node>(width / 2).strength(0.035))
      .force("y", d3.forceY<Node>(height * 0.42).strength(0.035))
      .force("groups", cohesion(groupHulls, byId))
      .force("collide", d3.forceCollide<Node>((n) => radius(n) + 22))
      .force("inertia-end", inertia(nodes, "end"))
      .alpha(0.6)
      .alphaDecay(0.015)
      .on("tick", ticked);
    ticked();
  }, [people, room, roomList, labels, relationships, hulls, genders, names, factors, size, onSelect]);

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
    edges.classed("lit", (e) => litRelationships.has(e.rel.id) || litPairs.has(pairKey(e.source.id, e.target.id)));
    const captioned = new Map<string, Edge>();
    edges.each((e) => {
      const key = pairKey(e.source.id, e.target.id);
      if (litPairs.get(key) && !captioned.has(key)) captioned.set(key, e);
    });
    const captions = svg
      .select("g.captions")
      .selectAll<SVGGElement, Edge>("g")
      .data([...captioned.values()], (e) => String(e.rel.id))
      .join((enter) => {
        const g = enter.append("g");
        g.append("rect").attr("rx", 10).attr("height", 20).attr("y", -10);
        g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em");
        return g;
      })
      .attr("transform", (e) => `translate(${control(e, 0.5)[0]},${control(e, 0.5)[1] + 16})`);
    captions.select("text").text((e) => shorten(litPairs.get(pairKey(e.source.id, e.target.id)) ?? ""));
    captions
      .select("rect")
      .attr("width", (e) => shorten(litPairs.get(pairKey(e.source.id, e.target.id)) ?? "").length * CHAR_PX + 20)
      .attr("x", (e) => -(shorten(litPairs.get(pairKey(e.source.id, e.target.id)) ?? "").length * CHAR_PX + 20) / 2);
    svg
      .select("g.nodes")
      .selectAll<SVGGElement, Node>("g.node")
      .classed("speaking", (n) => speakingId !== null && n.id === canonicalId(state, speakingId))
      .classed("mentioned", (n) => litPeople.has(n.id))
      .classed("selected", (n) => n.id === selectedId);
  }, [pulses, speakingId, selectedId, state, people, relationships, size]);

  return (
    <svg ref={svgRef} className="graph size-full" data-testid="graph">
      <defs>
        {Object.entries(GENDER_COLORS).map(([gender, [light, dark]]) => (
          <radialGradient key={gender} id={`node-${gender}`} cx="35%" cy="30%" r="75%">
            <stop offset="0%" stopColor={light} />
            <stop offset="100%" stopColor={dark} />
          </radialGradient>
        ))}
        <radialGradient id="graph-bg" cx="50%" cy="45%" r="75%">
          <stop offset="0%" stopColor="oklch(0.23 0.04 285)" />
          <stop offset="100%" stopColor="oklch(0.14 0.012 275)" />
        </radialGradient>
        <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
          <circle cx="1" cy="1" r="1" fill="oklch(1 0 0 / 7%)" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#graph-bg)" />
      <rect width="100%" height="100%" fill="url(#dots)" />
      <g className="layer">
        <g className="hulls" />
        <g className="edges" />
        <g className="edge-labels" />
        <g className="nodes" />
        <g className="captions" />
      </g>
    </svg>
  );
}

function shorten(text: string, max = 52): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
