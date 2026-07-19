import { useState } from "react";
import type { ReactNode } from "react";
import type { GraphDiff } from "../graph/diff";
import { diffCount } from "../graph/diff";
import { categoryColor } from "../graph/relationshipTypes";
import type { Person, Relationship, RelationshipCategory } from "../graph/types";

// One entry per process() request: the text sent + the actual graph diff it caused.
export interface LogEntry {
  id: number;
  text: string;
  time: string; // e.g. "14:03:22"
  diff: GraphDiff;
}

type Kind = "add" | "update" | "remove";

const KIND_GLYPH: Record<Kind, string> = { add: "＋", update: "～", remove: "－" };

function Swatch({ category }: { category: RelationshipCategory }) {
  return <span className="cl-swatch" style={{ background: categoryColor(category) }} title={category} />;
}

function Weight({ strength }: { strength?: number | null }) {
  if (strength == null) return null;
  return <span className="cl-weight">w{strength}</span>;
}

function personBody(p: Person): ReactNode {
  return (
    <>
      <span className="cl-verb">person</span>
      <span className="cl-name">{p.name}</span>
    </>
  );
}

function relBody(r: Relationship): ReactNode {
  return (
    <>
      <Swatch category={r.category} />
      <span className="cl-verb">{r.category}</span>
      <span className="cl-name">{r.source}</span>
      <span className="cl-arrow">{r.directed ? "→" : "—"}</span>
      <span className="cl-name">{r.target}</span>
      <span className="cl-label">{r.label}</span>
      <Weight strength={r.strength} />
    </>
  );
}

function relUpdateBody(before: Relationship, after: Relationship): ReactNode {
  const labelChanged = before.label !== after.label;
  const weightChanged = (before.strength ?? null) !== (after.strength ?? null);
  return (
    <>
      <Swatch category={after.category} />
      <span className="cl-verb">{after.category}</span>
      <span className="cl-name">{after.source}</span>
      <span className="cl-arrow">—</span>
      <span className="cl-name">{after.target}</span>
      {labelChanged ? (
        <span className="cl-label">
          {before.label} <span className="cl-arrow">→</span> {after.label}
        </span>
      ) : (
        <span className="cl-label">{after.label}</span>
      )}
      {weightChanged ? (
        <span className="cl-weight">
          w{before.strength ?? "?"}→{after.strength ?? "?"}
        </span>
      ) : (
        <Weight strength={after.strength} />
      )}
    </>
  );
}

function personUpdateBody(before: Person, after: Person): ReactNode {
  const addedAliases = after.aliases.filter((a) => !before.aliases.includes(a));
  const genderChanged = (before.gender ?? "unknown") !== (after.gender ?? "unknown");
  return (
    <>
      <span className="cl-verb">person</span>
      <span className="cl-name">{after.name}</span>
      {before.name !== after.name && <span className="cl-meta">was {before.name}</span>}
      {genderChanged && (
        <span className="cl-meta">
          gender: {before.gender ?? "unknown"} → {after.gender ?? "unknown"}
        </span>
      )}
      {addedAliases.length > 0 && <span className="cl-meta">+aliases: {addedAliases.join(", ")}</span>}
    </>
  );
}

interface Row {
  kind: Kind;
  body: ReactNode;
}

// Flatten a diff into ordered rows: adds, then updates, then removals.
function diffToRows(diff: GraphDiff): Row[] {
  const rows: Row[] = [];
  diff.peopleAdded.forEach((p) => rows.push({ kind: "add", body: personBody(p) }));
  diff.relsAdded.forEach((r) => rows.push({ kind: "add", body: relBody(r) }));
  diff.peopleUpdated.forEach(({ before, after }) =>
    rows.push({ kind: "update", body: personUpdateBody(before, after) }),
  );
  diff.relsUpdated.forEach(({ before, after }) =>
    rows.push({ kind: "update", body: relUpdateBody(before, after) }),
  );
  diff.relsRemoved.forEach((r) => rows.push({ kind: "remove", body: relBody(r) }));
  diff.peopleRemoved.forEach((p) => rows.push({ kind: "remove", body: personBody(p) }));
  return rows;
}

export function ChangeLog({ entries }: { entries: LogEntry[] }) {
  const [copiedId, setCopiedId] = useState<number | null>(null);

  async function copyText(entry: LogEntry) {
    try {
      await navigator.clipboard.writeText(entry.text);
      setCopiedId(entry.id);
      window.setTimeout(() => setCopiedId((id) => (id === entry.id ? null : id)), 1200);
    } catch {
      // clipboard unavailable (e.g. non-secure context) — silently no-op
    }
  }

  if (entries.length === 0) {
    return (
      <div className="cl">
        <div className="cl-title">Recent changes</div>
        <div className="cl-empty">Nothing yet.</div>
      </div>
    );
  }

  return (
    <div className="cl">
      <div className="cl-title">Recent changes</div>
      <div className="cl-entries">
        {entries.map((entry, i) => {
          const rows = diffToRows(entry.diff);
          const count = diffCount(entry.diff);
          return (
            <details key={entry.id} className="cl-entry" open={i === 0}>
            <summary className="cl-summary">
              <span className="cl-chevron" aria-hidden>
                ▸
              </span>
              <span className="cl-req" title={entry.text}>
                {entry.text}
              </span>
              <span className="cl-count">
                {count || "no"} {count === 1 ? "change" : "changes"}
              </span>
              <span className="cl-time">{entry.time}</span>
              <button
                className="cl-copy"
                onClick={(e) => {
                  e.preventDefault(); // don't toggle the <details>
                  e.stopPropagation();
                  void copyText(entry);
                }}
                title="Copy the original text input"
              >
                {copiedId === entry.id ? "Copied" : "Copy"}
              </button>
            </summary>
            <div className="cl-ops">
              {rows.length === 0 ? (
                <div className="cl-none">No graph changes for this input.</div>
              ) : (
                rows.map((row, j) => (
                  <div key={j} className={`cl-op cl-op-${row.kind}`}>
                    <span className="cl-glyph">{KIND_GLYPH[row.kind]}</span>
                    <span className="cl-op-body">{row.body}</span>
                  </div>
                ))
              )}
            </div>
          </details>
          );
        })}
      </div>
    </div>
  );
}
