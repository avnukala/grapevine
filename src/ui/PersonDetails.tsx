import type { Person, Relationship, SocialGraph } from "../graph/types";
import { categoryColor } from "../graph/relationshipTypes";

function RelRow({ rel, personId, graph }: { rel: Relationship; personId: string; graph: SocialGraph }) {
  const otherId = rel.source === personId ? rel.target : rel.source;
  const otherName = graph.people[otherId]?.name ?? otherId;

  // Direction from this person's point of view.
  let arrow = "—";
  if (rel.directed) arrow = rel.source === personId ? "→" : "←";

  return (
    <div className="pd-rel">
      <div className="pd-rel-top">
        <span className="pd-swatch" style={{ background: categoryColor(rel.category) }} />
        <span className="pd-arrow">{arrow}</span>
        <span className="pd-other">{otherName}</span>
        <span className="pd-cat">{rel.category}</span>
        {rel.strength != null && (
          <span className={`pd-weight${rel.strength < 0 ? " pd-weight-neg" : ""}`}>w{rel.strength}</span>
        )}
      </div>
      <div className="pd-label">{rel.label}</div>
      {rel.description && <div className="pd-desc">{rel.description}</div>}
    </div>
  );
}

export function PersonDetails({
  person,
  graph,
  onClose,
}: {
  person: Person;
  graph: SocialGraph;
  onClose: () => void;
}) {
  const rels = Object.values(graph.relationships).filter(
    (r) => r.source === person.id || r.target === person.id,
  );
  const attrs = person.attributes ?? {};

  return (
    <div className="pd-backdrop" onClick={onClose}>
      <div className="pd-card" onClick={(e) => e.stopPropagation()}>
        <div className="pd-header">
          <div>
            <div className="pd-name">{person.name}</div>
            {person.aliases.length > 0 && (
              <div className="pd-aliases">a.k.a. {person.aliases.join(", ")}</div>
            )}
          </div>
          <button className="pd-close" onClick={onClose} title="Close">
            ×
          </button>
        </div>

        {(attrs.role || attrs.org || attrs.notes) && (
          <div className="pd-attrs">
            {attrs.role && <span className="pd-attr">{attrs.role}</span>}
            {attrs.org && <span className="pd-attr">{attrs.org}</span>}
            {attrs.notes && <div className="pd-notes">{attrs.notes}</div>}
          </div>
        )}

        <div className="pd-section-title">
          {rels.length} {rels.length === 1 ? "relationship" : "relationships"}
        </div>

        <div className="pd-rels">
          {rels.length === 0 ? (
            <div className="pd-empty">No relationships yet.</div>
          ) : (
            rels.map((r) => <RelRow key={r.id} rel={r} personId={person.id} graph={graph} />)
          )}
        </div>
      </div>
    </div>
  );
}
