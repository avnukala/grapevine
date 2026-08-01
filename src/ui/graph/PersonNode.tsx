import { memo } from "react";
import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { Person } from "../../graph/types";
import { genderColor } from "../../graph/relationshipTypes";
import { personInitials } from "./geometry";
import { useFocus } from "./focus";

export interface PersonNodeData extends Record<string, unknown> {
  person: Person;
  flash: "add" | "update" | null;
}

export type PersonNodeType = Node<PersonNodeData, "person">;

function PersonNodeImpl({ id, data }: NodeProps<PersonNodeType>) {
  const { hoverId, neighborhood, selectedId } = useFocus();
  const { person, flash } = data;

  const isHovered = hoverId === id;
  const isSelected = selectedId === id;
  const dimmed = hoverId !== null && !neighborhood.has(id);

  const { role, org } = person.attributes ?? {};
  const showDetail = isHovered && (role || org);

  return (
    <div
      className={[
        "rf-pill",
        isSelected ? "rf-pill--selected" : "",
        dimmed ? "rf-pill--dimmed" : "",
        flash ? `rf-pill--flash-${flash}` : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0, pointerEvents: "none" }} />
      <div className="rf-pill-row">
        <span className="rf-badge" style={{ background: genderColor(person.gender) }}>
          {personInitials(person)}
        </span>
        <span className="rf-name">{person.name}</span>
      </div>
      {showDetail && (
        <div className="rf-detail">
          {[role, org].filter(Boolean).join(" · ")}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, pointerEvents: "none" }} />
    </div>
  );
}

// Position-only ticks (60/sec) must not re-render the pill's inner DOM — memo
// combined with keeping `data` identity stable across ticks (see
// useForceSimulation) is what keeps this cheap.
export const PersonNode = memo(PersonNodeImpl);
