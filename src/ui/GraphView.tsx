import { useEffect, useRef } from "react";
import cytoscape from "cytoscape";
import type { Core, ElementDefinition } from "cytoscape";
import type { SocialGraph } from "../graph/types";
import { categoryColor } from "../graph/relationshipTypes";

// Pure projection: SocialGraph -> Cytoscape elements.
function toElements(graph: SocialGraph): ElementDefinition[] {
  const nodes: ElementDefinition[] = Object.values(graph.people).map((p) => ({
    data: { id: p.id, label: p.name },
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
      "background-color": "#4c6ef5",
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
  {
    selector: "node.hl",
    style: {
      "background-color": "#3b5bdb",
      "border-color": "#dbe4ff",
      "overlay-color": "#4c6ef5",
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

export function GraphView({
  graph,
  onSelectPerson,
}: {
  graph: SocialGraph;
  onSelectPerson?: (id: string | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
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
      layout: { name: "cose", animate: true, padding: 40, nodeRepulsion: () => 9000 },
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

    return () => cy.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconcile elements whenever the graph changes, then re-run layout.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.json({ elements: toElements(graph) });
    cy.layout({ name: "cose", animate: true, padding: 40, nodeRepulsion: () => 9000 }).run();
  }, [graph]);

  return <div ref={containerRef} className="graph-canvas" />;
}
