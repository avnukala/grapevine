import { useState } from "react";
import { personName, roomIds } from "../state/derive";
import type { RunState, Segment } from "../state/types";

interface TranscriptProps {
  state: RunState;
  voiceThreshold: number;
  limit?: number;
  currentMs?: number | null;
  showTimings?: boolean;
  onAssign?: (segment: Segment, answer: { person_id: number } | { new_name: string }) => void;
  onSeek?: (ms: number) => void;
}

export function isUnknown(segment: Segment, threshold: number): boolean {
  return segment.person_id === null || (segment.speaker_confidence !== null && segment.speaker_confidence < threshold);
}

function clock(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

export function Transcript({ state, voiceThreshold, limit, currentMs, showTimings, onAssign, onSeek }: TranscriptProps) {
  const segments = Object.values(state.segments).sort((x, y) => x.start_ms - y.start_ms);
  const shown = limit ? segments.slice(-limit) : segments;
  return (
    <ol className="transcript" data-testid="transcript">
      {shown.map((segment) => {
        const unknown = isUnknown(segment, voiceThreshold);
        const current = currentMs !== undefined && currentMs !== null && segment.start_ms <= currentMs && currentMs < segment.end_ms;
        return (
          <li key={segment.id} className={`${unknown ? "unknown" : ""}${current ? " current" : ""}`}>
            <button className="time link" onClick={() => onSeek?.(segment.start_ms)} disabled={!onSeek}>
              {clock(segment.start_ms)}
            </button>
            <span className="speaker">
              {unknown ? `Unknown speaker (${segment.speaker_label})` : personName(state, segment.person_id)}
              {segment.speaker_confidence !== null && (
                <span className="confidence"> {Math.round(segment.speaker_confidence * 100)}%</span>
              )}
            </span>
            <span className="text">{segment.text}</span>
            {unknown && onAssign && <WhoSpoke state={state} onPick={(answer) => onAssign(segment, answer)} />}
            {showTimings && (
              <span className="timings">
                {Object.entries(segment.timings)
                  .map(([step, ms]) => `${step.replace("_ms", "")} ${ms}ms`)
                  .join(" · ")}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

interface WhoSpokeProps {
  state: RunState;
  onPick: (answer: { person_id: number } | { new_name: string }) => void;
}

function WhoSpoke({ state, onPick }: WhoSpokeProps) {
  const [name, setName] = useState("");
  return (
    <span className="who-spoke">
      Who spoke?
      {roomIds(state).map((id) => (
        <button key={id} onClick={() => onPick({ person_id: id })}>
          {personName(state, id).split(" ")[0]}
        </button>
      ))}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (name.trim()) onPick({ new_name: name.trim() });
        }}
      >
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New name" />
      </form>
    </span>
  );
}
