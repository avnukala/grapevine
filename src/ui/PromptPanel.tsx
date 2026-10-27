import { ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Prompt, RunState } from "../state/types";
import { PromptCard, QUESTIONS } from "./PromptCard";
import type { Answer } from "./PromptCard";

const MAX_SHOWN = 2;

interface PromptPanelProps {
  state: RunState;
  className?: string;
  hidden: boolean;
  onHidden: (hidden: boolean) => void;
  onAnswer: (prompt: Prompt, answer: Answer) => void;
  onDismiss: (prompt: Prompt) => void;
}

// Questions the room owes an answer come first, then the newest suggestions. Never more than two.
export function PromptPanel({ state, className, hidden, onHidden, onAnswer, onDismiss }: PromptPanelProps) {
  const open = Object.values(state.prompts).filter((p) => p.dismissed_at_ms === null);
  const questions = open.filter((p) => QUESTIONS.includes(p.reason)).sort((x, y) => x.id - y.id);
  const suggestions = open.filter((p) => !QUESTIONS.includes(p.reason)).sort((x, y) => y.id - x.id);
  const shown = [...questions, ...suggestions].slice(0, MAX_SHOWN);
  if (shown.length === 0) return null;
  return (
    <div className={cn("absolute top-4 right-4 z-10 flex w-[360px] flex-col gap-3", className)} data-testid="prompt-panel">
      <Button
        size="xs"
        variant="outline"
        className="self-end bg-card"
        aria-expanded={!hidden}
        onClick={() => onHidden(!hidden)}
        data-testid="prompt-toggle"
      >
        {hidden ? <ChevronDown /> : <ChevronUp />}
        {open.length === 1 ? "1 question" : `${open.length} questions`}
        {!hidden && open.length > shown.length && <span className="text-muted-foreground">· {shown.length} shown</span>}
      </Button>
      {!hidden &&
        shown.map((prompt) => (
          <PromptCard key={prompt.id} prompt={prompt} state={state} onAnswer={onAnswer} onDismiss={onDismiss} />
        ))}
    </div>
  );
}
