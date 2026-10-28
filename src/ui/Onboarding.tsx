import { LoaderCircle, Mic, Plus, Sparkles, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "../api";
import { GROUP_COLORS, initials } from "../state/categories";

interface OnboardingProps {
  blocked: boolean;
  known: string[];
  onCreated: (sessionId: number, runId: number | null) => Promise<void>;
  onCancel: (() => void) | null;
}

export function Onboarding({ blocked, known, onCreated, onCancel }: OnboardingProps) {
  const [title, setTitle] = useState("");
  const [names, setNames] = useState(["", ""]);
  const [words, setWords] = useState("");
  const [busy, setBusy] = useState<"live" | "save" | null>(null);
  const inputs = useRef<(HTMLInputElement | null)[]>([]);

  const roster = names.map((n) => n.trim()).filter(Boolean);
  const firstNames = roster.map((n) => n.split(/\s+/)[0].toLowerCase());
  const unclear = roster.some((n, i) => !n.includes(" ") && firstNames.some((f, j) => j !== i && f === firstNames[i]));
  const suggested = known.filter((n) => !roster.some((r) => r.toLowerCase() === n.toLowerCase())).slice(0, 12);

  // Fills the empty rows first, so a click on a suggestion or a pasted list does not leave gaps.
  function addNames(added: string[]) {
    setNames((current) => {
      const kept = current.filter((n) => n.trim());
      const next = [...kept, ...added];
      return next.length < 2 ? [...next, ""] : next;
    });
  }

  function addRow(after: number) {
    setNames((current) => [...current.slice(0, after + 1), "", ...current.slice(after + 1)]);
    requestAnimationFrame(() => inputs.current[after + 1]?.focus());
  }

  async function create(live: boolean) {
    setBusy(live ? "live" : "save");
    try {
      const name = title.trim() || `Session ${new Date().toLocaleDateString()}`;
      const vocabulary = words.split(",").map((w) => w.trim()).filter(Boolean);
      const session = await api.newSession({ name, roster, vocabulary });
      const run = live ? await api.startRun({ session_id: session.id, source: "mic" }) : null;
      await onCreated(session.id, run?.id ?? null);
    } catch (e) {
      toast.error(String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="flex h-full items-start justify-center overflow-auto p-8" data-testid="onboarding">
      <Card className="w-full max-w-xl border-border/60 bg-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <Sparkles className="size-5 text-primary" /> Who is here?
          </CardTitle>
          <CardDescription>
            Type the first and last name of each person. Grapevine uses this list to spell names correctly and to match
            each voice to the right person.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <Input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Name the session, for example Friday dinner at Maya's"
            aria-label="Session name"
          />

          <div className="space-y-2">
            {names.map((name, index) => (
              <div key={index} className="flex items-center gap-2">
                <span
                  className="flex size-8 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-ink transition-opacity"
                  style={{
                    background: GROUP_COLORS[index % GROUP_COLORS.length],
                    opacity: name.trim() ? 1 : 0.25,
                  }}
                >
                  {name.trim() ? initials(name.trim()) : index + 1}
                </span>
                <Input
                  ref={(el) => {
                    inputs.current[index] = el;
                  }}
                  value={name}
                  onChange={(e) => setNames((current) => current.map((n, i) => (i === index ? e.target.value : n)))}
                  onKeyDown={(e) => {
                    if (e.key !== "Enter") return;
                    e.preventDefault();
                    addRow(index);
                  }}
                  onPaste={(e) => {
                    const pasted = e.clipboardData.getData("text").split(/[,\n]/).map((n) => n.trim()).filter(Boolean);
                    if (pasted.length < 2) return;
                    e.preventDefault();
                    addNames(pasted);
                  }}
                  placeholder="First and last name"
                  aria-label={`Person ${index + 1}`}
                />
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={`Remove person ${index + 1}`}
                  disabled={names.length === 1}
                  onClick={() => setNames((current) => current.filter((_, i) => i !== index))}
                >
                  <X />
                </Button>
              </div>
            ))}
            <Button size="sm" variant="ghost" onClick={() => addRow(names.length - 1)}>
              <Plus /> Add person
            </Button>
            {suggested.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 pt-1 text-xs text-muted-foreground">
                In earlier sessions:
                {suggested.map((name) => (
                  <Button key={name} size="xs" variant="secondary" onClick={() => addNames([name])}>
                    <Plus /> {name}
                  </Button>
                ))}
              </div>
            )}
            {unclear && (
              <p className="text-xs text-warn">
                Two people have the same first name. Last names help tell them apart.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Input
              value={words}
              onChange={(e) => setWords(e.target.value)}
              placeholder="Words to know, such as EZoo, Glow, or a nickname"
              aria-label="Words to know"
            />
            <p className="text-xs text-muted-foreground">
              Separate them with commas. Grapevine spells these words the way you type them.
            </p>
          </div>

          <ol className="space-y-1.5 rounded-lg border border-border/60 bg-background/40 p-3 text-sm text-muted-foreground">
            <li>
              <span className="font-semibold text-foreground">1.</span> Start live. Grapevine listens through the
              microphone of this computer.
            </li>
            <li>
              <span className="font-semibold text-foreground">2.</span> Go around the room. Each person says "I'm" and
              their name, then one or two sentences about themselves.
            </li>
            <li>
              <span className="font-semibold text-foreground">3.</span> When everyone has spoken, click Done with
              introductions. Then talk as usual.
            </li>
          </ol>
        </CardContent>
        <CardFooter className="justify-end gap-2">
          {busy === "live" && (
            <span className="mr-auto text-xs text-muted-foreground">Loading the speech models. This can take a minute.</span>
          )}
          {onCancel && (
            <Button variant="ghost" onClick={onCancel} disabled={busy !== null}>
              Cancel
            </Button>
          )}
          <Button variant="outline" disabled={busy !== null || roster.length === 0} onClick={() => create(false)}>
            Save for later
          </Button>
          <Button disabled={busy !== null || blocked || roster.length === 0} onClick={() => create(true)}>
            {busy === "live" ? <LoaderCircle className="animate-spin" /> : <Mic />}
            {roster.length === 0
              ? "Start live"
              : `Start live with ${roster.length} ${roster.length === 1 ? "person" : "people"}`}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
