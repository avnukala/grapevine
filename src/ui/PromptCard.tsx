import { useState } from "react";
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

  return (
    <div className={`prompt-card reason-${prompt.reason}`} data-testid="prompt">
      <div className="prompt-text">{prompt.text}</div>
      <div className="prompt-actions">
        {(prompt.reason === "which_person" || prompt.reason === "which_speaker") && (
          <>
            {prompt.person_ids.map((id) => (
              <button key={id} onClick={() => onAnswer(prompt, { person_id: id })}>
                {personName(state, id)}
              </button>
            ))}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!name.trim()) return;
                onAnswer(
                  prompt,
                  prompt.reason === "which_person" ? { new_name: name.trim() } : { nickname: name.trim() },
                );
              }}
            >
              <input
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
              <button key={id} onClick={() => onAnswer(prompt, { keep_fact_id: id })} title={factText(id)}>
                Keep {personName(state, state.facts[id]?.told_by_person_id).split(" ")[0]}'s
              </button>
            ))}
            <button className="secondary" onClick={() => onAnswer(prompt, { keep_fact_id: null })}>
              Keep both
            </button>
          </>
        )}
        {prompt.reason === "merge_conflict" && (
          <>
            {factIds.map((id) => (
              <button key={id} onClick={() => onAnswer(prompt, { action: "keep", fact_id: id })} title={factText(id)}>
                Keep “{factText(id).slice(0, 40)}”
              </button>
            ))}
            <button className="secondary" onClick={() => onAnswer(prompt, { action: "keep_both" })}>
              Keep both as disputed
            </button>
            {((prompt.payload.relationship_ids as number[] | undefined) ?? []).length > 0 && (
              <button className="secondary" onClick={() => onAnswer(prompt, { action: "remove_edge" })}>
                Remove edge
              </button>
            )}
          </>
        )}
        {prompt.reason === "merge_people" && (
          <>
            <button onClick={() => onAnswer(prompt, { action: "merge_people" })}>Same person</button>
            <button className="secondary" onClick={() => onAnswer(prompt, { action: "keep_apart" })}>
              Different people
            </button>
          </>
        )}
        {!QUESTIONS.includes(prompt.reason) && !prompt.reason.startsWith("merge_") && (
          <button className="link" onClick={() => onDismiss(prompt)} aria-label="Dismiss">
            Dismiss
          </button>
        )}
      </div>
    </div>
  );
}
