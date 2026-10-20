import { DEFAULT_FACTORS } from "../state/derive";
import type { WeightFactors } from "../state/derive";

const LABELS: Record<keyof WeightFactors, string> = {
  yearsKnown: "Years known",
  recency: "Recency",
  closeness: "Closeness",
  sentiment: "Story tone",
  category: "Category",
};

const KEY = "grapevine.weights.v1";

export function loadFactors(): WeightFactors {
  const saved = localStorage.getItem(KEY);
  return saved ? { ...DEFAULT_FACTORS, ...(JSON.parse(saved) as Partial<WeightFactors>) } : DEFAULT_FACTORS;
}

export function saveFactors(factors: WeightFactors): void {
  localStorage.setItem(KEY, JSON.stringify(factors));
}

interface WeightPanelProps {
  factors: WeightFactors;
  onChange: (factors: WeightFactors) => void;
}

// Edge weight drives the graph physics. Tune the share of each input and compare the graphs.
export function WeightPanel({ factors, onChange }: WeightPanelProps) {
  const total = Object.values(factors).reduce((sum, w) => sum + w, 0) || 1;
  return (
    <div className="weights" data-testid="weights">
      {(Object.keys(LABELS) as (keyof WeightFactors)[]).map((key) => (
        <label key={key}>
          <span>{LABELS[key]}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={factors[key]}
            onChange={(e) => onChange({ ...factors, [key]: Number(e.target.value) })}
          />
          <span className="share">{Math.round((factors[key] / total) * 100)}%</span>
        </label>
      ))}
      <button className="link" onClick={() => onChange(DEFAULT_FACTORS)}>
        Reset
      </button>
    </div>
  );
}
