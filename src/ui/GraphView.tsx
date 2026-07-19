import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";
import type { SocialGraph } from "../graph/types";
import { categoryColor } from "../graph/relationshipTypes";
import { GraphPhysics, nodeRadius } from "./physics";

// All node movement is owned by GraphPhysics (src/ui/physics.ts): one continuous
// force simulation where positive edge weights pull pairs together and negative
// weights genuinely push them apart. Cytoscape is purely the renderer here —
// its own layouts are never run (elements mount via "preset" and the sim writes
// positions every tick).

// Node color by gender: male blue, female pink, ambiguous/unknown gray.
const GENDER_COLOR: Record<string, string> = {
  male: "#4c6ef5",
  female: "#e64980",
  unknown: "#868e96",
};
function genderColor(gender?: string): string {
  return GENDER_COLOR[gender ?? "unknown"] ?? GENDER_COLOR.unknown;
}

// Pure projection: SocialGraph -> Cytoscape elements.
function toElements(graph: SocialGraph): ElementDefinition[] {
  // Distinct-partner degree drives node size, so well-connected people read as hubs.
  const partners = new Map<string, Set<string>>();
  const touch = (a: string, b: string) => {
    let s = partners.get(a);
    if (!s) partners.set(a, (s = new Set()));
    s.add(b);
  };
  for (const r of Object.values(graph.relationships)) {
    touch(r.source, r.target);
    touch(r.target, r.source);
  }

  const nodes: ElementDefinition[] = Object.values(graph.people).map((p) => ({
    data: {
      id: p.id,
      label: p.name,
      color: genderColor(p.gender),
      size: Math.round(nodeRadius(partners.get(p.id)?.size ?? 0) * 2),
    },
  }));
  const edges: ElementDefinition[] = Object.values(graph.relationships).map((r) => {
    const strength = r.strength ?? 3; // signed affinity, default "ordinary"
    return {
      data: {
        id: r.id,
        source: r.source,
        target: r.target,
        label: r.label,
        color: categoryColor(r.category),
        arrow: r.directed ? "triangle" : "none",
        // line thickness reflects magnitude of the bond (positive or negative)
        width: Math.max(1.2, 1 + 0.75 * Math.abs(strength)),
        weight: strength, // signed; drives the layout physics
        // negative (antagonistic) relationships are drawn dashed
        lineStyle: strength < 0 ? "dashed" : "solid",
      },
    };
  });
  return [...nodes, ...edges];
}

// Cast helper for the few style values whose Cytoscape typings are stricter than
// the runtime (data-mappers, numeric transition durations).
const s = (v: unknown) => v as never;

const STYLESHEET: cytoscape.StylesheetStyle[] = [
  {
    selector: "node",
    style: {
      "background-color": "data(color)",
      "background-opacity": 0.95,
      "border-width": 2,
      "border-color": "#ffffff",
      "border-opacity": 1,
      label: "data(label)",
      color: "#1a1a2e",
      "font-size": "11px",
      "font-weight": s(500),
      "text-valign": "bottom",
      "text-halign": "center",
      "text-margin-y": 6,
      width: s("data(size)"),
      height: s("data(size)"),
      // smooth reactions to hover/selection state changes
      "transition-property": "background-color, border-color, border-width, opacity",
      "transition-duration": s(180),
      "transition-timing-function": "ease-out",
    },
  },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      "line-color": "data(color)",
      "line-style": s("data(lineStyle)"), // dashed = negative/antagonistic
      "line-opacity": s(0.55),
      "target-arrow-color": "data(color)",
      "target-arrow-shape": s("data(arrow)"),
      "arrow-scale": s(0.9),
      "curve-style": "bezier",
      label: "data(label)",
      "font-size": "9px",
      color: "#868e96",
      "text-opacity": s(0), // labels hidden until hover/selection
      // short relationship terms show in full; wrap to a 2nd line rather than truncate
      "text-wrap": s("wrap"),
      "text-max-width": s("110px"),
      "text-rotation": s("autorotate"),
      "text-background-color": "#ffffff",
      "text-background-opacity": 0.9,
      "text-background-padding": "3px",
      "text-background-shape": "roundrectangle",
      "transition-property": "line-opacity, width, text-opacity",
      "transition-duration": s(180),
      "transition-timing-function": "ease-out",
    },
  },
  // Soft halo + accent on the hovered/selected person.
  // Note: no background-color override here, so the gender color stays visible.
  {
    selector: "node.hl",
    style: {
      "border-color": "#dbe4ff",
      "overlay-color": "#868e96",
      "overlay-opacity": s(0.18),
      "overlay-padding": 10,
    },
  },
  // Edges connected to the focused node: brighter, thicker, labelled.
  {
    selector: "edge.hl",
    style: {
      "line-opacity": s(0.95),
      "text-opacity": s(1),
    },
  },
  // Everything not in focus recedes.
  {
    selector: ".faded",
    style: {
      opacity: s(0.12),
    },
  },
];

// Flash colors mirror the change-log's add / update language.
const FLASH_ADD = "#2f9e44";
const FLASH_UPDATE = "#f08c00";

const avg = (ns: number[]) => ns.reduce((a, b) => a + b, 0) / ns.length;

// Seed a new node at the centroid of its already-placed neighbors so it eases in
// from a sensible spot instead of flying in from the origin; fall back to the
// current viewport center when it has no placed neighbor yet.
function seedPosition(cy: Core, id: string, graph: SocialGraph) {
  const jitter = () => (Math.random() - 0.5) * 60;
  const placed = Object.values(graph.relationships)
    .filter((r) => r.source === id || r.target === id)
    .map((r) => cy.getElementById(r.source === id ? r.target : r.source))
    .filter((n) => n.nonempty() && n.isNode());
  if (placed.length > 0) {
    return {
      x: avg(placed.map((n) => n.position("x"))) + jitter(),
      y: avg(placed.map((n) => n.position("y"))) + jitter(),
    };
  }
  const e = cy.extent();
  return { x: (e.x1 + e.x2) / 2 + jitter(), y: (e.y1 + e.y2) / 2 + jitter() };
}

// A brief colored halo that blooms then fades — draws the eye to a just-changed
// element. Uses overlay-* so it never collides with the element's base style,
// and stop() first so rapid successive changes restart cleanly.
function bloom(el: cytoscape.CollectionReturnValue, color: string) {
  el.stop();
  el.style({ "overlay-color": color, "overlay-padding": el.isNode() ? 10 : 6, "overlay-opacity": 0.45 });
  el.animate({
    style: { "overlay-opacity": 0 },
    duration: 1000,
    easing: "ease-out",
    complete: () => el.removeStyle("overlay-color overlay-padding overlay-opacity"),
  });
}

export function GraphView({
  graph,
  onSelectPerson,
}: {
  graph: SocialGraph;
  onSelectPerson?: (id: string | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const physicsRef = useRef<GraphPhysics | null>(null);
  // keep the latest callback available to the one-time-bound cytoscape handlers
  const selectRef = useRef(onSelectPerson);
  selectRef.current = onSelectPerson;

  // Initialize Cytoscape (renderer) + GraphPhysics (simulation) once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: toElements(graph),
      style: STYLESHEET,
      layout: { name: "preset" }, // positions come from the simulation
      minZoom: 0.3,
      maxZoom: 2.5,
      wheelSensitivity: 0.2,
    });
    cyRef.current = cy;
    // Dev-only handle so browser automation / debugging can inspect positions.
    if (import.meta.env.DEV) (window as unknown as { cy?: Core }).cy = cy;

    const physics = new GraphPhysics(cy);
    physicsRef.current = physics;
    // Hydration (e.g. a session restore): settle fully off-screen, then frame it.
    physics.sync(graph); // no seeds — d3 spreads fresh nodes on its spiral
    if (cy.nodes().length > 0) {
      physics.settleNow();
      cy.fit(undefined, 60);
    }

    // Reactive focus: hovering a person highlights its neighborhood, fades the rest.
    const focus = (node: cytoscape.NodeSingular) => {
      const neighborhood = node.closedNeighborhood(); // the node + its edges + adjacent nodes
      cy.elements().addClass("faded");
      neighborhood.removeClass("faded");
      node.addClass("hl");
      node.connectedEdges().removeClass("faded").addClass("hl");
    };
    const clearFocus = () => {
      cy.elements().removeClass("faded hl");
    };

    cy.on("mouseover", "node", (e) => focus(e.target));
    cy.on("mouseout", "node", clearFocus);
    // Click a person -> open their relationship detail popup.
    cy.on("tap", "node", (e) => selectRef.current?.(e.target.id()));
    // Tapping empty space clears focus + closes the popup.
    cy.on("tap", (e) => {
      if (e.target === cy) {
        clearFocus();
        selectRef.current?.(null);
      }
    });
    // Pointer cursor over nodes.
    cy.on("mouseover", "node", () => {
      if (containerRef.current) containerRef.current.style.cursor = "pointer";
    });
    cy.on("mouseout", "node", () => {
      if (containerRef.current) containerRef.current.style.cursor = "default";
    });

    // Springy-on-touch: grab pins the node (a plain click stays perfectly
    // still); actually dragging wakes the simulation so friends follow and
    // rivals scatter live, with the same forces that shaped the layout. Release
    // lets it cool back to a freeze.
    cy.on("grab", "node", (e) => physics.grab(e.target.id()));
    cy.on("drag", "node", (e) => {
      const p = e.target.position();
      physics.drag(e.target.id(), p.x, p.y);
    });
    cy.on("free", "node", (e) => physics.free(e.target.id()));

    return () => {
      physics.destroy();
      physicsRef.current = null;
      cy.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconcile elements incrementally so the graph stays stable and feels reactive:
  //  - update the mutable display fields of existing elements in place
  //  - add new elements, seeding nodes near their neighbors so they ease in
  //  - re-point an edge whose endpoints flipped (same id, swapped source/target)
  //  - drop elements that are gone
  // Then hand the new structure to the simulation. Because the sim is already at
  // equilibrium, a reheat only moves what the change actually affects: the
  // newcomer settles in, a re-weighted pair drifts closer or springs apart, and
  // the rest of the graph barely stirs. Every added / updated element briefly
  // flashes so the eye can follow what the model just did.
  useEffect(() => {
    const cy = cyRef.current;
    const physics = physicsRef.current;
    if (!cy || !physics) return;

    const next = toElements(graph);
    const nextIds = new Set(next.map((d) => d.data.id as string));
    const wasEmpty = cy.nodes().length === 0;

    const added: string[] = [];
    const updated: string[] = [];
    let newNode = false;
    let physicsDirty = false; // anything structural: add/remove/flip or weight change

    cy.batch(() => {
      // Drop elements no longer present (removing a node cascades to its edges).
      cy.elements().forEach((el) => {
        if (!nextIds.has(el.id())) {
          physicsDirty = true;
          el.remove();
        }
      });

      for (const def of next) {
        const id = def.data.id as string;
        const isEdge = "source" in def.data;
        const ex = cy.getElementById(id);

        // Brand-new element: seed nodes near their neighbors, add edges directly.
        if (!ex.nonempty()) {
          if (isEdge) {
            cy.add({ group: "edges", data: def.data });
            physicsDirty = true;
          } else {
            cy.add({ group: "nodes", data: def.data, position: seedPosition(cy, id, graph) });
            newNode = true;
          }
          added.push(id);
          continue;
        }

        // An edge keeps its id but can flip source/target (the id is the sorted
        // pair). Cytoscape can't repoint an edge, so replace it in place.
        if (isEdge && (ex.data("source") !== def.data.source || ex.data("target") !== def.data.target)) {
          ex.remove();
          cy.add({ group: "edges", data: def.data });
          updated.push(id);
          physicsDirty = true;
          continue;
        }

        // Existing element: sync only the fields that drive its appearance.
        const fields = isEdge
          ? (["label", "color", "arrow", "width", "weight", "lineStyle"] as const)
          : (["label", "color", "size"] as const);
        let changed = false;
        for (const k of fields) {
          if (ex.data(k) !== def.data[k]) {
            ex.data(k, def.data[k]);
            // A size bump is a side effect of a new edge, not news — don't flash.
            if (k !== "size") changed = true;
            if (k === "weight") physicsDirty = true; // re-settle with the new force
          }
        }
        if (changed) updated.push(id);
      }
    });

    // Flash after the batch so halos render against the settled graph.
    added.forEach((id) => bloom(cy.getElementById(id), FLASH_ADD));
    updated.forEach((id) => bloom(cy.getElementById(id), FLASH_UPDATE));

    if (!newNode && !physicsDirty) return; // cosmetic change only — leave the sim asleep

    // New nodes enter the sim at their seeded spot (neighbor centroid).
    physics.sync(graph, (id) => ({ ...cy.getElementById(id).position() }));
    if (wasEmpty && cy.nodes().length > 0) {
      // First fill: settle instantly and frame the result.
      physics.settleNow();
      cy.fit(undefined, 60);
    } else {
      // Otherwise animate: a bigger kick when someone new must find their place,
      // a smaller one when only the forces changed.
      physics.reheat(newNode ? 0.65 : 0.45);
    }
  }, [graph]);

  return <div ref={containerRef} className="graph-canvas" />;
}
