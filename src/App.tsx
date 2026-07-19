import { useCallback, useEffect, useRef, useState } from "react";
import { emptyGraph } from "./graph/types";
import type { SocialGraph } from "./graph/types";
import { applyOps } from "./graph/reducer";
import { diffGraphs } from "./graph/diff";
import { extractOps } from "./llm/client";
import { GraphView } from "./ui/GraphView";
import { PersonDetails } from "./ui/PersonDetails";
import { ChangeLog } from "./ui/ChangeLog";
import type { LogEntry } from "./ui/ChangeLog";
import { RELATIONSHIP_CATEGORIES, CATEGORY_LIST } from "./graph/relationshipTypes";
import { loadState, saveState, clearState } from "./persist";

// How much recent transcript to carry forward as reference context (chars).
const CONTEXT_MAX = 400;

export default function App() {
  // Hydrate from sessionStorage so a normal reload restores the working state.
  const [graph, setGraph] = useState<SocialGraph>(() => loadState()?.graph ?? emptyGraph());
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<LogEntry[]>(() => loadState()?.entries ?? []);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  // Rolling window of recently-processed text, sent as reference context so the
  // model can resolve pronouns / unnamed subjects carried over from prior input.
  const [recentContext, setRecentContext] = useState<string>(() => loadState()?.recentContext ?? "");

  // Undo stack: graph snapshots taken *before* each applied request, newest first.
  // Stays index-aligned with `entries` so undo can drop both together.
  const [past, setPast] = useState<SocialGraph[]>(() => loadState()?.past ?? []);
  const pastRef = useRef(past);
  pastRef.current = past;

  // Persist on every change to graph / log / undo stack / context window.
  useEffect(() => {
    saveState({ graph, entries, past, recentContext });
  }, [graph, entries, past, recentContext]);

  async function process() {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    const requestText = text.trim();
    const snapshot = graph; // state to return to if this request is undone
    try {
      // Send the prior window as reference context; it's already in the graph.
      const ops = await extractOps(graph, requestText, recentContext);
      const nextGraph = applyOps(snapshot, ops);
      setGraph(nextGraph);
      const entry: LogEntry = {
        id: Date.now(),
        text: requestText,
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
        // Log the actual graph diff (captures cascade removals, drops no-ops).
        diff: diffGraphs(snapshot, nextGraph),
      };
      setEntries((prev) => [entry, ...prev]);
      setPast((prev) => [snapshot, ...prev]);
      // Extend the rolling context window, keeping the most recent tail.
      setRecentContext((prev) => `${prev}\n${requestText}`.slice(-CONTEXT_MAX));
      setText("");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // Revert the most recent request: restore the pre-request graph and drop its log entry.
  const undo = useCallback(() => {
    const p = pastRef.current;
    if (p.length === 0) return;
    setGraph(p[0]);
    setPast(p.slice(1));
    setEntries((prev) => prev.slice(1));
  }, []);

  // Deliberate wipe: clear graph, log, undo stack, and the persisted copy.
  function reset() {
    if (!window.confirm("Clear the entire graph and history? This can't be undone.")) return;
    setGraph(emptyGraph());
    setEntries([]);
    setPast([]);
    setRecentContext("");
    clearState();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    // Cmd/Ctrl+Enter to process.
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void process();
    }
  }

  // Global ⌘/Ctrl+Z for graph undo — but not while typing, so the textarea's
  // native text-undo keeps working.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && (e.key === "z" || e.key === "Z")) {
        const el = document.activeElement;
        const typing = el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement;
        if (typing) return;
        e.preventDefault();
        undo();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo]);

  const peopleCount = Object.keys(graph.people).length;
  const relCount = Object.keys(graph.relationships).length;
  const canUndo = past.length > 0;

  return (
    <div className="app">
      <aside className="panel">
        <h1>🍇 Grapevine</h1>
        <p className="sub">
          Describe people and how they know each other. Each pass sends the text plus the current
          graph to the model, which returns incremental operations.
        </p>

        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            "e.g. Sarah and Mike are dating. Sarah has been best friends with Priya since childhood. Mike reports to Dana at work."
          }
          rows={6}
        />
        <div className="toolbar">
          <button onClick={() => void process()} disabled={busy || !text.trim()}>
            {busy ? "Processing…" : "Process (⌘/Ctrl+Enter)"}
          </button>
          <button
            className="secondary"
            onClick={undo}
            disabled={!canUndo}
            title="Undo the most recent request (⌘/Ctrl+Z)"
          >
            ↶ Undo{canUndo ? ` (${past.length})` : ""}
          </button>
          <button
            className="secondary"
            onClick={reset}
            disabled={peopleCount === 0 && entries.length === 0}
            title="Clear the graph and history"
          >
            Reset
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="stats">
          {peopleCount} {peopleCount === 1 ? "person" : "people"} · {relCount}{" "}
          {relCount === 1 ? "relationship" : "relationships"}
        </div>

        <div className="legend">
          {CATEGORY_LIST.map((c) => (
            <span key={c} className="legend-item">
              <span className="swatch" style={{ background: RELATIONSHIP_CATEGORIES[c].color }} />
              {c}
            </span>
          ))}
        </div>

        <ChangeLog entries={entries} />
      </aside>

      <main className="canvas-wrap">
        <GraphView graph={graph} onSelectPerson={setSelectedId} />
        {selectedId && graph.people[selectedId] && (
          <PersonDetails
            person={graph.people[selectedId]}
            graph={graph}
            onClose={() => setSelectedId(null)}
          />
        )}
      </main>
    </div>
  );
}
