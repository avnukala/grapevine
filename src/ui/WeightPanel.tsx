import { SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
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
    <Popover>
      <PopoverTrigger asChild>
        <Button size="sm" variant="outline">
          <SlidersHorizontal /> Edge weight
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80" data-testid="weights">
        <div className="mb-3 text-sm font-medium">Edge weight</div>
        <div className="space-y-4">
          {(Object.keys(LABELS) as (keyof WeightFactors)[]).map((key) => (
            <div key={key} className="grid grid-cols-[96px_1fr_40px] items-center gap-3 text-sm">
              <span className="text-muted-foreground">{LABELS[key]}</span>
              <Slider
                min={0}
                max={1}
                aria-label={LABELS[key]}
                step={0.05}
                value={[factors[key]]}
                onValueChange={([value]) => onChange({ ...factors, [key]: value })}
              />
              <span className="text-right tabular-nums">{Math.round((factors[key] / total) * 100)}%</span>
            </div>
          ))}
        </div>
        <Button className="mt-4" size="sm" variant="ghost" onClick={() => onChange(DEFAULT_FACTORS)}>
          Reset
        </Button>
      </PopoverContent>
    </Popover>
  );
}
