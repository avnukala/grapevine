import { BaseEdge, EdgeLabelRenderer, useStore, type EdgeProps, type Edge, type InternalNode } from "@xyflow/react";
import type { Relationship } from "../../graph/types";
import { getEdgeParams, parallelOffset } from "./geometry";
import { useFocus } from "./focus";

export interface RelationshipEdgeData extends Record<string, unknown> {
  relationship: Relationship;
  color: string;
  flash: "add" | "update" | null;
  parallelIndex: number;
  parallelCount: number;
}

export type RelationshipEdgeType = Edge<RelationshipEdgeData, "relationship">;

export function RelationshipEdge({ id, source, target, data, markerEnd }: EdgeProps<RelationshipEdgeType>) {
  const sourceNode = useStore((s) => s.nodeLookup.get(source) as InternalNode | undefined);
  const targetNode = useStore((s) => s.nodeLookup.get(target) as InternalNode | undefined);
  const { hoverId, neighborhood } = useFocus();

  if (!sourceNode || !targetNode || !data) return null;

  const { relationship, color, flash, parallelIndex, parallelCount } = data;
  const { sx, sy, tx, ty } = getEdgeParams(sourceNode, targetNode);
  const { mx, my } = parallelOffset(sx, sy, tx, ty, parallelIndex, parallelCount);
  const path = `M ${sx},${sy} Q ${mx},${my} ${tx},${ty}`;

  const focused = hoverId !== null && neighborhood.has(source) && neighborhood.has(target);
  const dimmed = hoverId !== null && !focused;

  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{
          stroke: color,
          strokeWidth: 1.5 + ((relationship.strength ?? 3) - 1),
          opacity: dimmed ? 0.08 : focused ? 0.95 : 0.55,
          transition: "opacity 180ms ease-out",
        }}
        className={flash ? `rf-edge--flash-${flash}` : undefined}
      />
      <EdgeLabelRenderer>
        <div
          className={["rf-edge-label", focused ? "rf-edge-label--visible" : ""].filter(Boolean).join(" ")}
          style={{
            transform: `translate(-50%, -50%) translate(${mx}px, ${my}px)`,
          }}
        >
          {relationship.label}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}
