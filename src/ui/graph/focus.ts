import { createContext, useContext } from "react";

// Hover focus lives outside node/edge `data` so it never breaks the position-only
// object identity the tick loop relies on (see useForceSimulation). Nodes/edges
// read this via useContext, which intentionally bypasses React.memo.
export interface FocusState {
  hoverId: string | null;
  neighborhood: Set<string>; // hovered person + directly connected people
  selectedId: string | null; // person whose detail popup is open
}

export const FocusContext = createContext<FocusState>({
  hoverId: null,
  neighborhood: new Set(),
  selectedId: null,
});

export function useFocus(): FocusState {
  return useContext(FocusContext);
}
