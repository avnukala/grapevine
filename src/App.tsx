import { useEffect, useState } from "react";
import type { WeightFactors } from "./state/derive";
import { DebugScreen } from "./ui/DebugScreen";
import { GlobalScreen } from "./ui/GlobalScreen";
import { LiveScreen } from "./ui/LiveScreen";
import { loadFactors, saveFactors } from "./ui/WeightPanel";

type Route = "live" | "global" | "debug";

// The debug screen exists only in dev and test builds, so the live view never shows it.
const DEBUG = import.meta.env.DEV || import.meta.env.MODE === "test";

function route(): Route {
  const name = location.hash.replace(/^#\/?/, "").split("?")[0];
  if (name === "global") return "global";
  if (name === "debug" && DEBUG) return "debug";
  return "live";
}

export default function App() {
  const [current, setCurrent] = useState<Route>(route);
  const [factors, setFactors] = useState<WeightFactors>(loadFactors);

  useEffect(() => {
    const onHash = () => setCurrent(route());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const onFactors = (next: WeightFactors) => {
    setFactors(next);
    saveFactors(next);
  };

  return (
    <div className="app">
      <nav className="topbar">
        <h1>Grapevine</h1>
        <a href="#/" className={current === "live" ? "on" : ""}>
          Live
        </a>
        <a href="#/global" className={current === "global" ? "on" : ""}>
          Global graph
        </a>
        {DEBUG && (
          <a href="#/debug" className={current === "debug" ? "on" : ""}>
            Debug and replay
          </a>
        )}
      </nav>
      {current === "live" && <LiveScreen factors={factors} onFactors={onFactors} />}
      {current === "global" && <GlobalScreen factors={factors} />}
      {current === "debug" && <DebugScreen factors={factors} />}
    </div>
  );
}
