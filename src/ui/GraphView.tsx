import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  useReactFlow,
  useNodesState,
  useEdgesState,
  type XYPosition,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { SocialGraph } from "../graph/types";
import { diffGraphs } from "../graph/diff";
import { categoryColor } from "../graph/relationshipTypes";
import { PersonNode, type PersonNodeType } from "./graph/PersonNode";
import { RelationshipEdge, type RelationshipEdgeType } from "./graph/RelationshipEdge";
import { FocusContext, type FocusState } from "./graph/focus";
import { estimateNodeSize } from "./graph/geometry";
import { useForceSimulation } from "./graph/useForceSimulation";

// Module-level constants — declaring these inline would remount every node on
// every render, which would be catastrophic at 60fps tick rate.
const NODE_TYPES = { person: PersonNode };
const EDGE_TYPES = { relationship: RelationshipEdge };

const FLASH_MS = 1000;

function buildEdgeKey(a: string, b: string): string {
  return [a, b].sort().join("__");
}

function GraphViewInner({
  graph,
  onSelectPerson,
}: {
  graph: SocialGraph;
  onSelectPerson?: (id: string | null) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<PersonNodeType>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<RelationshipEdgeType>([]);
  const { fitView } = useReactFlow();

  const [hoverId, setHoverId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [flashes, setFlashes] = useState<Map<string, "add" | "update">>(new Map());

  const prevGraphRef = useRef<SocialGraph>(graph);
  const flashTimersRef = useRef<Map<string, number>>(new Map());
  const draggingRef = useRef<string | null>(null);

  // Neighborhood of the hovered person, for dim/highlight — recomputed only
  // when hover changes, never touches node/edge `data`.
  const neighborhood = useMemo(() => {
    const set = new Set<string>();
    if (!hoverId) return set;
    set.add(hoverId);
    for (const r of Object.values(graph.relationships)) {
      if (r.source === hoverId) set.add(r.target);
      if (r.target === hoverId) set.add(r.source);
    }
    return set;
  }, [graph, hoverId]);

  const focusValue = useMemo<FocusState>(
    () => ({ hoverId, neighborhood, selectedId }),
    [hoverId, neighborhood, selectedId],
  );

  // Push position updates into React Flow. d3-force works on centres; React
  // Flow's `position` is the top-left corner — this is the single boundary
  // where that conversion happens.
  const onTick = useCallback(
    (positions: Map<string, XYPosition>) => {
      setNodes((prev) =>
        prev.map((n) => {
          if (n.id === draggingRef.current) return n; // React Flow owns the dragged node
          const c = positions.get(n.id);
          if (!c) return n;
          const w = n.measured?.width ?? n.width ?? estimateNodeSize(n.data.person).width;
          const h = n.measured?.height ?? n.height ?? estimateNodeSize(n.data.person).height;
          const x = c.x - w / 2;
          const y = c.y - h / 2;
          if (n.position.x === x && n.position.y === y) return n;
          return { ...n, position: { x, y } };
        }),
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [setNodes],
  );

  const { sync, pin, release } = useForceSimulation({ onTick, draggingRef });

  // Incremental reconciliation: diff against the previous graph, add/update
  // nodes+edges, seed new nodes near their neighbors via the sim, flash changes.
  useEffect(() => {
    const prev = prevGraphRef.current;
    const diff = diffGraphs(prev, graph);
    prevGraphRef.current = graph;

    const wasEmpty = Object.keys(prev.people).length === 0;
    const addedIds = new Set(diff.peopleAdded.map((p) => p.id));

    const rect = containerRef.current?.getBoundingClientRect();
    const center = rect ? { x: rect.width / 2, y: rect.height / 2 } : { x: 0, y: 0 };

    sync(graph, addedIds, center);

    // Build the next node list, preserving existing RF positions (the sim's
    // tick loop is the only thing that ever moves settled nodes).
    setNodes((prevNodes) => {
      const byId = new Map(prevNodes.map((n) => [n.id, n]));
      return Object.values(graph.people).map((p) => {
        const size = estimateNodeSize(p);
        const existing = byId.get(p.id);
        const flash = flashes.get(p.id) ?? null;
        if (existing) {
          return {
            ...existing,
            width: size.width,
            height: size.height,
            data: { person: p, flash },
          };
        }
        return {
          id: p.id,
          type: "person" as const,
          position: { x: center.x - size.width / 2, y: center.y - size.height / 2 },
          width: size.width,
          height: size.height,
          data: { person: p, flash },
        };
      });
    });

    // Parallel-edge indexing: same pair can hold one edge per category
    // (relationshipId in reducer.ts), so group by unordered pair before assigning
    // { parallelIndex, parallelCount }.
    const byPair = new Map<string, string[]>();
    for (const r of Object.values(graph.relationships)) {
      const key = buildEdgeKey(r.source, r.target);
      const list = byPair.get(key) ?? [];
      list.push(r.id);
      byPair.set(key, list);
    }

    setEdges(() =>
      Object.values(graph.relationships).map((r) => {
        const pairIds = byPair.get(buildEdgeKey(r.source, r.target))!;
        const flash = flashes.get(r.id) ?? null;
        return {
          id: r.id,
          type: "relationship" as const,
          source: r.source,
          target: r.target,
          markerEnd: r.directed ? { type: MarkerType.ArrowClosed, color: categoryColor(r.category) } : undefined,
          data: {
            relationship: r,
            color: categoryColor(r.category),
            flash,
            parallelIndex: pairIds.indexOf(r.id),
            parallelCount: pairIds.length,
          },
        };
      }),
    );

    // Flash newly added / updated elements, then clear after FLASH_MS.
    const changedAdd = [...diff.peopleAdded.map((p) => p.id), ...diff.relsAdded.map((r) => r.id)];
    const changedUpdate = [
      ...diff.peopleUpdated.map((u) => u.after.id),
      ...diff.relsUpdated.map((u) => u.after.id),
    ];
    if (changedAdd.length > 0 || changedUpdate.length > 0) {
      setFlashes((prevFlashes) => {
        const next = new Map(prevFlashes);
        for (const id of changedAdd) next.set(id, "add");
        for (const id of changedUpdate) next.set(id, "update");
        return next;
      });
      for (const id of [...changedAdd, ...changedUpdate]) {
        window.clearTimeout(flashTimersRef.current.get(id));
        const timer = window.setTimeout(() => {
          setFlashes((prevFlashes) => {
            const next = new Map(prevFlashes);
            next.delete(id);
            return next;
          });
          flashTimersRef.current.delete(id);
        }, FLASH_MS);
        flashTimersRef.current.set(id, timer);
      }
    }

    if (wasEmpty && Object.keys(graph.people).length > 0) {
      requestAnimationFrame(() => fitView({ padding: 0.4, duration: 300 }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  // Apply flash state to already-rendered nodes/edges without a full rebuild.
  useEffect(() => {
    setNodes((prev) =>
      prev.map((n) => {
        const flash = flashes.get(n.id) ?? null;
        if (n.data.flash === flash) return n;
        return { ...n, data: { ...n.data, flash } };
      }),
    );
    setEdges((prev) =>
      prev.map((e) => {
        const flash = flashes.get(e.id) ?? null;
        if (e.data?.flash === flash) return e;
        return { ...e, data: { ...e.data!, flash } };
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flashes]);

  // Clear flash timers on unmount.
  useEffect(() => {
    return () => {
      for (const t of flashTimersRef.current.values()) window.clearTimeout(t);
    };
  }, []);

  const onNodeDrag = useCallback(
    (_: unknown, node: { id: string; position: XYPosition; measured?: { width?: number; height?: number } }) => {
      // Wake physics on drag (not on grab/click) — a plain click must stay calm.
      const w = node.measured?.width ?? 0;
      const h = node.measured?.height ?? 0;
      pin(node.id, { x: node.position.x + w / 2, y: node.position.y + h / 2 });
    },
    [pin],
  );

  const onNodeDragStop = useCallback(
    (_: unknown, node: { id: string }) => {
      release(node.id);
    },
    [release],
  );

  return (
    <div ref={containerRef} className="graph-canvas">
      <FocusContext.Provider value={focusValue}>
        <ReactFlow<PersonNodeType, RelationshipEdgeType>
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          minZoom={0.3}
          maxZoom={2.5}
          nodesConnectable={false}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onNodeMouseEnter={(_, n) => setHoverId(n.id)}
          onNodeMouseLeave={() => setHoverId(null)}
          onNodeClick={(_, n) => {
            setSelectedId(n.id);
            onSelectPerson?.(n.id);
          }}
          onPaneClick={() => {
            setSelectedId(null);
            setHoverId(null);
            onSelectPerson?.(null);
          }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#e9ecef" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </FocusContext.Provider>
    </div>
  );
}

export function GraphView(props: { graph: SocialGraph; onSelectPerson?: (id: string | null) => void }) {
  return (
    <ReactFlowProvider>
      <GraphViewInner {...props} />
    </ReactFlowProvider>
  );
}
