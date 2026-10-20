import type { Prompt, RunState } from "../state/types";
import { PromptCard, QUESTIONS } from "./PromptCard";
import type { Answer } from "./PromptCard";

const MAX_SHOWN = 2;

interface PromptPanelProps {
  state: RunState;
  onAnswer: (prompt: Prompt, answer: Answer) => void;
  onDismiss: (prompt: Prompt) => void;
}

// Questions the room owes an answer come first, then the newest suggestions. Never more than two.
export function PromptPanel({ state, onAnswer, onDismiss }: PromptPanelProps) {
  const open = Object.values(state.prompts).filter((p) => p.dismissed_at_ms === null);
  const questions = open.filter((p) => QUESTIONS.includes(p.reason)).sort((x, y) => x.id - y.id);
  const suggestions = open.filter((p) => !QUESTIONS.includes(p.reason)).sort((x, y) => y.id - x.id);
  const shown = [...questions, ...suggestions].slice(0, MAX_SHOWN);
  if (shown.length === 0) return null;
  return (
    <div className="prompt-panel" data-testid="prompt-panel">
      {shown.map((prompt) => (
        <PromptCard key={prompt.id} prompt={prompt} state={state} onAnswer={onAnswer} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
