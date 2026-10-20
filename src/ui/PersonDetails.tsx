import { CATEGORIES } from "../state/categories";
import { canonicalId, liveFacts, personName } from "../state/derive";
import type { RunState } from "../state/types";

interface PersonDetailsProps {
  state: RunState;
  personId: number;
  onClose: () => void;
}

export function PersonDetails({ state, personId, onClose }: PersonDetailsProps) {
  const person = state.people[personId];
  if (!person) return null;
  const same = (id: number) => canonicalId(state, id) === personId;
  const relationships = Object.values(state.relationships).filter((r) => same(r.a) || same(r.b));
  const facts = liveFacts(state).filter((f) => f.people.some((p) => same(p.person_id) && p.role !== "teller"));
  const groups = Object.values(state.members)
    .filter((m) => same(m.person_id))
    .map((m) => state.groups[m.group_id]?.name)
    .filter(Boolean);

  return (
    <aside className="details" data-testid="person-details">
      <header>
        <h2>{person.name}</h2>
        <button className="link" onClick={onClose} aria-label="Close">
          Close
        </button>
      </header>
      {person.aliases.length > 0 && <p className="muted">Also called {person.aliases.join(", ")}</p>}
      {groups.length > 0 && <p className="muted">Groups: {[...new Set(groups)].join(", ")}</p>}
      <h3>Relationships</h3>
      <ul>
        {relationships.map((r) => (
          <li key={r.id}>
            <span className="swatch" style={{ background: CATEGORIES[r.category].color }} />
            <strong>{personName(state, same(r.a) ? r.b : r.a)}</strong>: {r.label}
            {r.status === "past" ? " (past)" : ""}
            {r.met_on ? `, met ${r.met_on}` : ""}
            {r.summary && <div className="muted">{r.summary}</div>}
          </li>
        ))}
        {relationships.length === 0 && <li className="muted">None yet</li>}
      </ul>
      <h3>Facts</h3>
      <ul>
        {facts.map((f) => (
          <li key={f.id} className={f.disputed ? "disputed" : ""}>
            {f.text}
            {f.start_date ? ` (${f.start_date})` : ""}
            {f.disputed ? " — disputed" : ""}
            {f.detail && <div className="muted">{f.detail}</div>}
          </li>
        ))}
        {facts.length === 0 && <li className="muted">None yet</li>}
      </ul>
    </aside>
  );
}
