import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import cola from "cytoscape-cola";
import type { Core, ElementDefinition } from "cytoscape";
import type { SocialGraph } from "../graph/types";
import { categoryColor } from "../graph/relationshipTypes";

// Register cola so we can run its continuous physics simulation on demand.
cytoscape.use(cola);

// ---- Physics tuning ----

// One-shot force params for the structural cose passes (initial + incremental):
// strong repulsion + overlap avoidance so people claim their own space, springy
// edges, light gravity to keep the graph cohesive, and gaps between components.
const COSE_PHYSICS = {
  nodeRepulsion: () => 20000,
  nodeOverlap: 24,
  idealEdgeLength: () => 90,
  edgeElasticity: () => 100,
  gravity: 0.3,
  componentSpacing: 120,
};

// Continuous physics, run only while a node is being dragged (see wakePhysics).
// infinite:true keeps it ticking; centerGraph/fit:false so waking never recenters
// or zooms the graph. cola pins grabbed + locked nodes internally each tick.
const COLA_OPTIONS = {
  name: "cola",
  infinite: true,
  fit: false,
  centerGraph: false,
  randomize: false,
  ungrabifyWhileSimulating: false,
  avoidOverlap: true,
  handleDisconnected: true,
  nodeSpacing: () => 12,
  edgeLength: 100,
} as unknown as cytoscape.LayoutOptions;

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
  const nodes: ElementDefinition[] = Object.values(graph.people).map((p) => ({
    data: { id: p.id, label: p.name, color: genderColor(p.gender) },
  }));
  const edges: ElementDefinition[] = Object.values(graph.relationships).map((r) => {
    const strength = r.strength ?? 3; // default to "ordinary" if unset
    return {
      data: {
        id: r.id,
        source: r.source,
        target: r.target,
        label: r.label,
        color: categoryColor(r.category),
        arrow: r.directed ? "triangle" : "none",
        // map weight 1..5 -> a slim line width 1.5..5.5px
        width: 1.5 + (strength - 1),
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
      width: 26,
      height: 26,
      // smooth reactions to hover/selection state changes
      "transition-property": "background-color, border-color, border-width, width, height, opacity",
      "transition-duration": s(180),
      "transition-timing-function": "ease-out",
    },
  },
  {
    selector: "edge",
    style: {
      width: "data(width)",
      "line-color": "data(color)",
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
  // The in-flight incremental layout, so a rapid follow-up run can stop it first.
  const layoutRef = useRef<cytoscape.Layouts | null>(null);
  // The live drag-physics simulation (cola), awake only while a node is dragged.
  const colaRef = useRef<cytoscape.Layouts | null>(null);
  const sleepTimerRef = useRef<number | undefined>(undefined);
  // keep the latest callback available to the one-time-bound cytoscape handlers
  const selectRef = useRef(onSelectPerson);
  selectRef.current = onSelectPerson;

  // Initialize Cytoscape once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      elements: toElements(graph),
      style: STYLESHEET,
      layout: { name: "cose", animate: true, padding: 40, ...COSE_PHYSICS },
      minZoom: 0.3,
      maxZoom: 2.5,
      wheelSensitivity: 0.2,
    });
    cyRef.current = cy;

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

    // Springy-on-touch: the graph is frozen at rest, but dragging a node wakes a
    // live cola simulation so neighbors repel / spring away in response. It sleeps
    // shortly after release, refreezing the layout. Waking on "drag" (not "grab")
    // means a plain click still just opens the details popup without any jiggle.
    const wakePhysics = () => {
      window.clearTimeout(sleepTimerRef.current);
      if (colaRef.current) return; // already awake
      layoutRef.current?.stop(); // don't let a cose settle fight the sim
      cy.nodes().unlock(); // everything is free to respond while dragging
      const sim = cy.layout(COLA_OPTIONS);
      colaRef.current = sim;
      sim.run();
    };
    const sleepPhysics = () => {
      window.clearTimeout(sleepTimerRef.current);
      sleepTimerRef.current = window.setTimeout(() => {
        colaRef.current?.stop();
        colaRef.current = null;
      }, 900);
    };
    cy.on("drag", "node", wakePhysics);
    cy.on("free", "node", sleepPhysics);

    return () => {
      window.clearTimeout(sleepTimerRef.current);
      cy.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconcile elements incrementally so the graph stays stable and feels reactive:
  //  - update the mutable display fields of existing elements in place
  //  - add new elements, seeding nodes near their neighbors so they ease in
  //  - re-point an edge whose endpoints flipped (same id, swapped source/target)
  //  - drop elements that are gone
  // When a node is added, the established graph is locked so it stays put while
  // only the newcomer settles into place; edge-only and removal-only changes skip
  // layout entirely. Every added / updated element briefly flashes so the eye can
  // follow what the model just did.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;

    const next = toElements(graph);
    const nextIds = new Set(next.map((d) => d.data.id as string));
    const wasEmpty = cy.nodes().length === 0;

    const added: string[] = [];
    const updated: string[] = [];
    let newNode = false;

    cy.batch(() => {
      // Drop elements no longer present (removing a node cascades to its edges).
      cy.elements().forEach((el) => {
        if (!nextIds.has(el.id())) el.remove();
      });

      for (const def of next) {
        const id = def.data.id as string;
        const isEdge = "source" in def.data;
        const ex = cy.getElementById(id);

        // Brand-new element: seed nodes near their neighbors, add edges directly.
        if (!ex.nonempty()) {
          if (isEdge) {
            cy.add({ group: "edges", data: def.data });
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
          continue;
        }

        // Existing element: sync only the fields that drive its appearance.
        const fields = isEdge ? (["label", "color", "arrow", "width"] as const) : (["label", "color"] as const);
        let changed = false;
        for (const k of fields) {
          if (ex.data(k) !== def.data[k]) {
            ex.data(k, def.data[k]);
            changed = true;
          }
        }
        if (changed) updated.push(id);
      }
    });

    // Flash after the batch so halos render against the settled graph.
    added.forEach((id) => bloom(cy.getElementById(id), FLASH_ADD));
    updated.forEach((id) => bloom(cy.getElementById(id), FLASH_UPDATE));

    // A new node needs placing. Lock the established graph so it stays fixed and
    // only the newcomer settles; re-fit the viewport only on the first fill.
    if (newNode) {
      layoutRef.current?.stop();
      colaRef.current?.stop(); // don't let the drag sim fight the structural pass
      colaRef.current = null;
      window.clearTimeout(sleepTimerRef.current);
      cy.nodes().unlock(); // clear any locks left by an interrupted prior run
      const addedSet = new Set(added);
      cy.nodes()
        .filter((n) => !addedSet.has(n.id()))
        .lock();
      const layout = cy.layout({
        name: "cose",
        animate: true,
        fit: wasEmpty,
        padding: 40,
        randomize: false,
        ...COSE_PHYSICS,
      });
      layoutRef.current = layout;
      // Only the most recent layout clears the locks — guards against a stale
      // stop() firing after a newer run has already started.
      layout.on("layoutstop", () => {
        if (layoutRef.current === layout) cy.nodes().unlock();
      });
      layout.run();
    }
  }, [graph]);

  return <div ref={containerRef} className="graph-canvas" />;
}
