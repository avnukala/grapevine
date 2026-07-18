import type { SocialGraph } from "./graph/types";
import type { LogEntry } from "./ui/ChangeLog";

// sessionStorage: survives normal reloads within the tab, clears when the tab is
// closed. (Note: a hard refresh keeps it too — browsers don't expose a way to
// detect a hard refresh, so use the Reset button for a deliberate wipe.)
const KEY = "grapevine.session.v1";

export interface PersistedState {
  graph: SocialGraph;
  entries: LogEntry[];
  past: SocialGraph[];
}

export function loadState(): PersistedState | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as PersistedState) : null;
  } catch {
    return null;
  }
}

export function saveState(state: PersistedState): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    // ignore quota / serialization errors
  }
}

export function clearState(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
