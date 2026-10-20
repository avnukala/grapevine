import * as d3 from "d3";
import { useEffect, useMemo, useRef, useState } from "react";
import { dateRange, liveFacts, personName } from "../state/derive";
import type { DateRange } from "../state/derive";
import type { Fact, GraphEvent, RunState } from "../state/types";

interface TimelineViewProps {
  state: RunState;
  // Fact and event ids that just changed. The newest one with a date moves into view.
  recentFactIds: number[];
  recentEventIds: number[];
}

interface Row {
  key: string;
  label: string;
  range: DateRange;
  event: GraphEvent | null;
  facts: { fact: Fact; range: DateRange }[];
}

const ROW_HEIGHT = 34;
const TOP = 36;
const LEFT = 16;
const RIGHT = 280;

function factRange(fact: Fact): DateRange | null {
  return dateRange(fact.start_date, fact.end_date, fact.date_precision);
}

// Only resolved dates show. A fact inside a dated event sits on the event's row.
function rows(state: RunState): Row[] {
  const facts = liveFacts(state);
  const result: Row[] = [];
  const placed = new Set<number>();
  for (const event of Object.values(state.events)) {
    const range = dateRange(event.start_date, event.end_date, event.date_precision);
    if (!range) continue;
    const own = facts.filter((f) => f.event_id === event.id);
    own.forEach((f) => placed.add(f.id));
    result.push({
      key: `e${event.id}`,
      label: event.name,
      range,
      event,
      facts: own.map((fact) => ({ fact, range: factRange(fact) ?? range })),
    });
  }
  for (const fact of facts) {
    const range = factRange(fact);
    if (placed.has(fact.id) || !range) continue;
    result.push({ key: `f${fact.id}`, label: fact.text, range, event: null, facts: [{ fact, range }] });
  }
  return result.sort((x, y) => x.range.start.getTime() - y.range.start.getTime());
}

export function TimelineView({ state, recentFactIds, recentEventIds }: TimelineViewProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(900);
  const [transform, setTransform] = useState(d3.zoomIdentity);
  const [hover, setHover] = useState<string | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const items = useMemo(() => rows(state), [state]);

  const base = useMemo(() => {
    const starts = items.map((r) => r.range.start);
    const ends = items.map((r) => r.range.end);
    const min = d3.min(starts) ?? new Date(2015, 0, 1);
    const max = d3.max(ends) ?? new Date();
    const pad = Math.max((max.getTime() - min.getTime()) * 0.08, 180 * 24 * 3600 * 1000);
    return d3
      .scaleTime()
      .domain([new Date(min.getTime() - pad), new Date(max.getTime() + pad)])
      .range([LEFT, Math.max(width - RIGHT, LEFT + 200)]);
  }, [items, width]);
  const x = transform.rescaleX(base);

  useEffect(() => {
    const svg = svgRef.current!;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(svg);
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 40])
      .on("zoom", (event) => setTransform(event.transform));
    zoomRef.current = zoom;
    d3.select(svg).call(zoom);
    return () => observer.disconnect();
  }, []);

  const recent = new Set([...recentFactIds.map((id) => `f${id}`), ...recentEventIds.map((id) => `e${id}`)]);
  const focus = [...items]
    .reverse()
    .find((row) => recent.has(row.key) || row.facts.some((f) => recent.has(`f${f.fact.id}`)));
  const focusKey = focus?.key ?? null;

  useEffect(() => {
    if (!focus || !zoomRef.current) return;
    const center = base((focus.range.start.getTime() + focus.range.end.getTime()) / 2);
    const k = transform.k;
    const target = d3.zoomIdentity.translate(width / 2 - center * k, 0).scale(k);
    d3.select(svgRef.current!).transition().duration(700).call(zoomRef.current.transform, target);
  }, [focusKey]); // Moves only when a new row comes into focus, not on every render.

  const height = Math.max(TOP + items.length * ROW_HEIGHT + 20, 200);
  const ticks = x.ticks(Math.max(2, Math.floor(width / 110)));

  return (
    <div className="timeline-wrap">
      <svg ref={svgRef} className="timeline" style={{ height }} data-testid="timeline">
        {ticks.map((tick) => (
          <g key={tick.getTime()} transform={`translate(${x(tick)},0)`}>
            <line y1={TOP - 6} y2={height} className="tick-line" />
            <text y={TOP - 12} textAnchor="middle" className="tick-label">
              {x.tickFormat()(tick)}
            </text>
          </g>
        ))}
        {items.map((row, i) => {
          const y = TOP + i * ROW_HEIGHT + ROW_HEIGHT / 2;
          const x0 = x(row.range.start);
          const x1 = Math.max(x(row.range.end), x0 + 6);
          const lit = recent.has(row.key) || row.facts.some((f) => recent.has(`f${f.fact.id}`));
          return (
            <g key={row.key} className={`timeline-row${lit ? " lit" : ""}`} data-testid="timeline-row">
              {row.event && <rect x={x0} y={y - 9} width={x1 - x0} height={18} rx={4} className="event-bar" />}
              {row.facts.map(({ fact, range }) => (
                <circle
                  key={fact.id}
                  cx={x((range.start.getTime() + range.end.getTime()) / 2)}
                  cy={y}
                  r={5}
                  className={`fact-dot${fact.disputed ? " disputed" : ""}`}
                  onMouseEnter={() => setHover(`${fact.text} — told by ${personName(state, fact.told_by_person_id)}`)}
                  onMouseLeave={() => setHover(null)}
                />
              ))}
              <text x={x1 + 8} y={y + 4} className={row.event ? "event-label" : "fact-label"}>
                {row.label}
                {row.event?.when_text ? ` (${row.event.when_text})` : ""}
              </text>
            </g>
          );
        })}
        {items.length === 0 && (
          <text x={width / 2} y={110} textAnchor="middle" className="empty">
            No dated events or facts yet
          </text>
        )}
      </svg>
      {hover && <div className="timeline-hover">{hover}</div>}
    </div>
  );
}
