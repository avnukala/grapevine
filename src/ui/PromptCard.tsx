import { HelpCircle, MessageCircleQuestion, Scale, Users, X } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { personName } from "../state/derive";
import type { Prompt, RunState } from "../state/types";

export type Answer =
  | { person_id: number }
  | { new_name: string }
  | { nickname: string }
  | { keep_fact_id: number | null }
  | { action: "keep"; fact_id: number }
  | { action: "keep_both" | "remove_edge" | "merge_people" | "keep_apart" };

export const QUESTIONS = ["which_person", "which_speaker", "conflict"];

const STYLE: Record<string, { icon: typeof HelpCircle; accent: string; title: string }> = {
  which_person: { icon: Users, accent: "border-l-amber-400", title: "Who do you mean?" },
  which_speaker: { icon: Users, accent: "border-l-amber-400", title: "Who is speaking?" },
  conflict: { icon: Scale, accent: "border-l-red-400", title: "Facts disagree" },
  merge_conflict: { icon: Scale, accent: "border-l-red-400", title: "Sessions disagree" },
  merge_people: { icon: Users, accent: "border-l-amber-400", title: "Same person?" },
};
const SUGGESTION = { icon: MessageCircleQuestion, accent: "border-l-violet-400", title: "Ask the room" };

interface PromptCardProps {
  prompt: Prompt;
  state: RunState;
  onAnswer: (prompt: Prompt, answer: Answer) => void;
  onDismiss: (prompt: Prompt) => void;
}

export function PromptCard({ prompt, state, onAnswer, onDismiss }: PromptCardProps) {
  const [name, setName] = useState("");
  const factIds = (prompt.payload.fact_ids as number[] | undefined) ?? [];
  const factText = (id: number) => state.facts[id]?.text ?? `Fact ${id}`;
  const style = STYLE[prompt.reason] ?? SUGGESTION;
  const Icon = style.icon;
  const suggestion = !(prompt.reason in STYLE);

  return (
    <div
      className={cn(
        "animate-in fade-in slide-in-from-right-4 rounded-xl border border-l-4 border-border bg-card/90 p-4 shadow-2xl backdrop-blur",
        style.accent,
      )}
      data-testid="prompt"
    >
      <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
        <Icon className="size-3.5" />
        {style.title}
        {suggestion && (
          <button className="ml-auto text-muted-foreground hover:text-foreground" onClick={() => onDismiss(prompt)} aria-label="Dismiss">
            <X className="size-4" />
          </button>
        )}
      </div>
      <p className="text-[15px] leading-snug">{prompt.text}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {(prompt.reason === "which_person" || prompt.reason === "which_speaker") && (
          <>
            {prompt.person_ids.map((id) => (
              <Button key={id} size="sm" onClick={() => onAnswer(prompt, { person_id: id })}>
                {personName(state, id)}
              </Button>
            ))}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!name.trim()) return;
                onAnswer(prompt, prompt.reason === "which_person" ? { new_name: name.trim() } : { nickname: name.trim() });
              }}
            >
              <Input
                className="h-8 w-36"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={prompt.reason === "which_person" ? "Someone else" : "Nickname"}
              />
            </form>
          </>
        )}
        {prompt.reason === "conflict" && (
          <>
            {factIds.map((id) => (
              <Button key={id} size="sm" onClick={() => onAnswer(prompt, { keep_fact_id: id })} title={factText(id)}>
                Keep {personName(state, state.facts[id]?.told_by_person_id).split(" ")[0]}'s
              </Button>
            ))}
            <Button size="sm" variant="outline" onClick={() => onAnswer(prompt, { keep_fact_id: null })}>
              Keep both
            </Button>
          </>
        )}
        {prompt.reason === "merge_conflict" && (
          <>
            {factIds.map((id) => (
              <Button
                key={id}
                size="sm"
                variant="secondary"
                className="h-auto max-w-full py-1.5 text-left whitespace-normal"
                onClick={() => onAnswer(prompt, { action: "keep", fact_id: id })}
              >
                Keep “{factText(id)}”
              </Button>
            ))}
            <Button size="sm" variant="outline" onClick={() => onAnswer(prompt, { action: "keep_both" })}>
              Keep both as disputed
            </Button>
            {((prompt.payload.relationship_ids as number[] | undefined) ?? []).length > 0 && (
              <Button size="sm" variant="destructive" onClick={() => onAnswer(prompt, { action: "remove_edge" })}>
                Remove edge
              </Button>
            )}
          </>
        )}
        {prompt.reason === "merge_people" && (
          <>
            <Button size="sm" onClick={() => onAnswer(prompt, { action: "merge_people" })}>
              Same person
            </Button>
            <Button size="sm" variant="outline" onClick={() => onAnswer(prompt, { action: "keep_apart" })}>
              Different people
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
