import { Grape, Moon, Sun } from "lucide-react";
import { ThemeProvider, useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
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

const LINKS: { route: Route; href: string; label: string }[] = [
  { route: "live", href: "#/", label: "Live" },
  { route: "global", href: "#/global", label: "Global graph" },
  ...(DEBUG ? [{ route: "debug" as Route, href: "#/debug", label: "Debug and replay" }] : []),
];

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

  // The client renders the theme script, so it never runs. A data type keeps React from warning about it.
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
      scriptProps={{ type: "application/json" }}
    >
      <TooltipProvider>
        <div className="flex h-svh flex-col">
          <header className="flex h-12 shrink-0 items-center gap-6 border-b border-border bg-card px-4">
            <div className="flex items-center gap-2 font-semibold tracking-tight">
              <span className="flex size-7 items-center justify-center rounded-md bg-primary">
                <Grape className="size-4 text-primary-foreground" />
              </span>
              Grapevine
            </div>
            <nav className="flex gap-1">
              {LINKS.map((link) => (
                <a
                  key={link.route}
                  href={link.href}
                  className={cn(
                    "rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:text-foreground",
                    current === link.route && "bg-secondary text-foreground",
                  )}
                >
                  {link.label}
                </a>
              ))}
            </nav>
            <ThemeToggle />
          </header>
          <main className="flex min-h-0 flex-1 flex-col">
            {current === "live" && <LiveScreen factors={factors} onFactors={onFactors} />}
            {current === "global" && <GlobalScreen factors={factors} />}
            {current === "debug" && <DebugScreen factors={factors} />}
          </main>
        </div>
        {/* Above the legend and the edge weight button in the bottom right of the graph. */}
        <Toaster position="bottom-right" offset={{ bottom: 100 }} />
      </TooltipProvider>
    </ThemeProvider>
  );
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const dark = resolvedTheme === "dark";
  return (
    <Button
      size="icon"
      variant="ghost"
      className="ml-auto"
      aria-label={dark ? "Use light mode" : "Use dark mode"}
      onClick={() => setTheme(dark ? "light" : "dark")}
    >
      {dark ? <Sun /> : <Moon />}
    </Button>
  );
}
