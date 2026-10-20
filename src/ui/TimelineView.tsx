import * as d3 from "d3";
import { CalendarDays, MapPin, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CATEGORIES, GENDER_COLORS, KIND_COLORS, initials } from "../state/categories";
import { canonicalId, dateRange, liveFacts, personName, roomIds } from "../state/derive";
import type { DateRange } from "../state/derive";
import type { Fact, GraphEvent, Relationship, RunState } from "../state/types";

interface TimelineViewProps {
  state: RunState;
  // Fact and event ids that just changed. The newest one with a date moves into view.
  recentFactIds: number[];
  recentEventIds: number[];
}

type Item =
  | { kind: "event"; key: string; event: GraphEvent; range: DateRange }
  | { kind: "fact"; key: string; fact: Fact; range: DateRange; lanes: number[] }
  | { kind: "met"; key: string; rel: Relationship; range: DateRange; lanes: number[] };

interface Lane {
  personId: number;
  name: string;
}

const LABEL_WIDTH = 168;
const AXIS_HEIGHT = 34;
const EVENT_ROW = 26;
const LANE_HEIGHT = 46;
const BRUSH_HEIGHT = 44;
const DAY_MS = 24 * 3600 * 1000;

function factRange(fact: Fact, state: RunState): DateRange | null {
  const own = dateRange(fact.start_date, fact.end_date, fact.date_precision);
  if (own) return own;
  const event = fact.event_id ? state.events[fact.event_id] : null;
  return event ? dateRange(event.start_date, event.end_date, event.date_precision) : null;
}

function mid(range: DateRange): Date {
  return new Date((range.start.getTime() + range.end.getTime()) / 2);
}

// Packs event bars into rows so that bars and their labels do not overlap at the given scale.
function packEvents(items: Item[], x: d3.ScaleTime<number, number>): Map<string, number> {
  const rowEnds: number[] = [];
  const rows = new Map<string, number>();
  for (const item of items) {
    if (item.kind !== "event") continue;
    const start = x(item.range.start);
    const end = Math.max(x(item.range.end), start + 8) + item.event.name.length * 6.5 + 20;
    let row = rowEnds.findIndex((rowEnd) => rowEnd < start);
    if (row === -1) row = rowEnds.length;
    rowEnds[row] = end;
    rows.set(item.key, row);
  }
  return rows;
}

function layout(state: RunState): { lanes: Lane[]; items: Item[] } {
  const events = Object.values(state.events)
    .map((event) => ({ event, range: dateRange(event.start_date, event.end_date, event.date_precision) }))
    .filter((e): e is { event: GraphEvent; range: DateRange } => e.range !== null)
    .sort((x, y) => x.range.start.getTime() - y.range.start.getTime());

  const facts = liveFacts(state)
    .map((fact) => ({ fact, range: factRange(fact, state) }))
    .filter((f): f is { fact: Fact; range: DateRange } => f.range !== null && !(f.fact.kind === "story" && f.fact.segment_ids.length === 0));
  const mets = Object.values(state.relationships)
    .map((rel) => ({ rel, range: rel.met_on ? dateRange(rel.met_on, null, null) : null }))
    .filter((m): m is { rel: Relationship; range: DateRange } => m.range !== null);

  const people = new Set<number>();
  facts.forEach(({ fact }) => fact.people.filter((p) => p.role !== "teller").forEach((p) => people.add(canonicalId(state, p.person_id))));
  mets.forEach(({ rel }) => [rel.a, rel.b].forEach((id) => people.add(canonicalId(state, id))));
  const room = roomIds(state).map((id) => canonicalId(state, id));
  const ordered = [...room.filter((id) => people.has(id)), ...[...people].filter((id) => !room.includes(id))];
  const lanes = ordered.map((personId) => ({ personId, name: personName(state, personId) }));
  const laneOf = new Map(lanes.map((lane, i) => [lane.personId, i]));

  const items: Item[] = [
    ...events.map(({ event, range }): Item => ({ kind: "event", key: `e${event.id}`, event, range })),
    ...facts.map(({ fact, range }): Item => ({
      kind: "fact",
      key: `f${fact.id}`,
      fact,
      range,
      lanes: [...new Set(fact.people.filter((p) => p.role !== "teller").map((p) => laneOf.get(canonicalId(state, p.person_id))!))],
    })),
    ...mets.map(({ rel, range }): Item => ({
      kind: "met",
      key: `r${rel.id}`,
      rel,
      range,
      lanes: [laneOf.get(canonicalId(state, rel.a))!, laneOf.get(canonicalId(state, rel.b))!],
    })),
  ];
  return { lanes, items };
}

export function TimelineView({ state, recentFactIds, recentEventIds }: TimelineViewProps) {
  const { lanes, items } = useMemo(() => layout(state), [state]);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const recent = new Set([...recentFactIds.map((id) => `f${id}`), ...recentEventIds.map((id) => `e${id}`)]);
  const newest = [...items].reverse().find((item) => recent.has(item.key))?.key ?? null;

  useEffect(() => {
    if (newest) setFocusKey(newest);
  }, [newest]);

  return (
    <div className="flex size-full min-h-0" data-testid="timeline">
      <SwimlaneChart
        state={state}
        lanes={lanes}
        items={items}
        recent={recent}
        focusKey={focusKey}
        onFocus={setFocusKey}
      />
      <StoryFeed state={state} items={items} recent={recent} focusKey={focusKey} onFocus={setFocusKey} />
    </div>
  );
}

interface SwimlaneChartProps {
  state: RunState;
  lanes: Lane[];
  items: Item[];
  recent: Set<string>;
  focusKey: string | null;
  onFocus: (key: string) => void;
}

interface Hover {
  x: number;
  y: number;
  item: Item;
}

function SwimlaneChart({ state, lanes, items, recent, focusKey, onFocus }: SwimlaneChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<SVGSVGElement>(null);
  const brushRef = useRef<SVGGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [width, setWidth] = useState(900);
  const [transform, setTransform] = useState(d3.zoomIdentity);
  const [hover, setHover] = useState<Hover | null>(null);

  const plotWidth = Math.max(width - LABEL_WIDTH - 24, 200);
  const base = useMemo(() => {
    const min = d3.min(items, (i) => i.range.start) ?? new Date(2015, 0, 1);
    const max = d3.max(items, (i) => i.range.end) ?? new Date();
    const pad = Math.max((max.getTime() - min.getTime()) * 0.06, 200 * DAY_MS);
    // Extra room on the right keeps the label of the last event inside the chart.
    return d3
      .scaleTime()
      .domain([new Date(min.getTime() - pad), new Date(max.getTime() + pad * 3)])
      .range([0, plotWidth]);
  }, [items, plotWidth]);
  const x = transform.rescaleX(base);
  const rows = useMemo(() => packEvents(items, base), [items, base]);
  const eventsHeight = Math.max(1, ...[...rows.values()].map((r) => r + 1)) * EVENT_ROW + 16;
  const lanesTop = AXIS_HEIGHT + eventsHeight;
  const height = lanesTop + Math.max(lanes.length, 1) * LANE_HEIGHT + 12;

  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(wrapRef.current!);
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 80])
      .on("zoom", (event) => setTransform(event.transform));
    zoomRef.current = zoom;
    d3.select(chartRef.current!).call(zoom).on("dblclick.zoom", null);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    zoomRef.current?.translateExtent([
      [0, 0],
      [plotWidth, height],
    ]).extent([
      [0, 0],
      [plotWidth, height],
    ]);
  }, [plotWidth, height]);

  // The minimap brush and the zoom show the same window. A brush move zooms the chart.
  useEffect(() => {
    const brush = d3
      .brushX()
      .extent([
        [0, 4],
        [plotWidth, BRUSH_HEIGHT - 4],
      ])
      .on("brush end", (event) => {
        if (!event.sourceEvent || !event.selection) return;
        const [x0, x1] = event.selection as [number, number];
        const k = plotWidth / Math.max(x1 - x0, 1);
        d3.select(chartRef.current!).call(zoomRef.current!.transform, d3.zoomIdentity.scale(k).translate(-x0, 0));
      });
    const g = d3.select(brushRef.current!);
    g.call(brush);
    const [d0, d1] = x.domain();
    g.call(brush.move, [Math.max(0, base(d0)), Math.min(plotWidth, base(d1))]);
  }, [plotWidth, base, transform]);

  useEffect(() => {
    const item = items.find((i) => i.key === focusKey);
    if (!item || !zoomRef.current) return;
    const k = Math.max(transform.k, 1);
    const target = d3.zoomIdentity.translate(plotWidth / 2 - base(mid(item.range)) * k, 0).scale(k);
    d3.select(chartRef.current!).transition().duration(750).call(zoomRef.current.transform, target);
  }, [focusKey]); // Moves only when focus changes, not on every render.

  const fit = () => d3.select(chartRef.current!).transition().duration(500).call(zoomRef.current!.transform, d3.zoomIdentity);
  const laneY = (lane: number) => lanesTop + lane * LANE_HEIGHT + LANE_HEIGHT / 2;
  const ticks = x.ticks(Math.max(2, Math.floor(plotWidth / 90)));
  const format = x.tickFormat();
  const show = (event: React.MouseEvent, item: Item) => {
    const box = wrapRef.current!.getBoundingClientRect();
    setHover({ x: event.clientX - box.left, y: event.clientY - box.top, item });
  };

  return (
    <div ref={wrapRef} className="relative min-w-0 flex-1 overflow-auto pt-14">
      <div className="absolute top-3 right-4 z-10 flex items-center gap-2">
        <Legend />
        <Button size="sm" variant="outline" onClick={fit}>
          Fit all
        </Button>
      </div>
      <svg width={width} height={height} className="block select-none">
        <g transform={`translate(${LABEL_WIDTH},0)`}>
          {lanes.map((lane, i) => (
            <rect
              key={lane.personId}
              x={-LABEL_WIDTH}
              y={lanesTop + i * LANE_HEIGHT}
              width={width}
              height={LANE_HEIGHT}
              className={i % 2 ? "fill-white/[0.015]" : "fill-white/[0.035]"}
            />
          ))}
          <rect x={-LABEL_WIDTH} y={AXIS_HEIGHT} width={width} height={eventsHeight} className="fill-violet-400/[0.05]" />
        </g>
        <g transform={`translate(${LABEL_WIDTH},0)`}>
          <svg ref={chartRef} width={plotWidth} height={height} className="cursor-grab overflow-hidden" data-testid="timeline-chart">
            <rect width={plotWidth} height={height} fill="transparent" />
            {ticks.map((tick) => (
              <g key={tick.getTime()} transform={`translate(${x(tick)},0)`}>
                <line y1={AXIS_HEIGHT - 4} y2={height} className="stroke-white/[0.07]" />
                <text y={AXIS_HEIGHT - 12} textAnchor="middle" className="fill-muted-foreground text-[11px]">
                  {format(tick)}
                </text>
              </g>
            ))}
            {items.map((item) => {
              const lit = recent.has(item.key) || focusKey === item.key;
              const cx = x(mid(item.range));
              switch (item.kind) {
                case "event": {
                  const x0 = x(item.range.start);
                  const w = Math.max(x(item.range.end) - x0, 8);
                  const y = AXIS_HEIGHT + 8 + (rows.get(item.key) ?? 0) * EVENT_ROW;
                  return (
                    <g
                      key={item.key}
                      className="cursor-pointer"
                      onMouseMove={(e) => show(e, item)}
                      onMouseLeave={() => setHover(null)}
                      onClick={() => onFocus(item.key)}
                      data-testid="timeline-event"
                    >
                      <rect
                        x={x0}
                        y={y}
                        width={w}
                        height={EVENT_ROW - 6}
                        rx={6}
                        fill="url(#event-fill)"
                        className={lit ? "stroke-amber-300 [filter:drop-shadow(0_0_8px_rgb(252_211_77))]" : "stroke-violet-300/50"}
                      />
                      <text
                        x={w > item.event.name.length * 6.5 + 16 ? x0 + 8 : x0 + w + 6}
                        y={y + 14}
                        className="fill-foreground text-[11px] font-semibold"
                      >
                        {item.event.name}
                      </text>
                    </g>
                  );
                }
                case "fact":
                  return (
                    <g
                      key={item.key}
                      className="cursor-pointer"
                      onMouseMove={(e) => show(e, item)}
                      onMouseLeave={() => setHover(null)}
                      onClick={() => onFocus(item.key)}
                      data-testid="timeline-fact"
                    >
                      {item.lanes.length > 1 && (
                        <line
                          x1={cx}
                          x2={cx}
                          y1={laneY(Math.min(...item.lanes))}
                          y2={laneY(Math.max(...item.lanes))}
                          stroke={KIND_COLORS[item.fact.kind]}
                          strokeOpacity={0.45}
                          strokeWidth={2}
                        />
                      )}
                      {item.lanes.map((lane) => (
                        <circle
                          key={lane}
                          cx={cx}
                          cy={laneY(lane)}
                          r={lit ? 8 : 6}
                          fill={KIND_COLORS[item.fact.kind]}
                          stroke={item.fact.disputed ? "#f87171" : "oklch(0.16 0.015 275)"}
                          strokeWidth={item.fact.disputed ? 3 : 2}
                          className={lit ? "[filter:drop-shadow(0_0_8px_rgb(252_211_77))]" : ""}
                        />
                      ))}
                    </g>
                  );
                case "met":
                  return (
                    <g
                      key={item.key}
                      className="cursor-pointer"
                      onMouseMove={(e) => show(e, item)}
                      onMouseLeave={() => setHover(null)}
                    >
                      <line
                        x1={cx}
                        x2={cx}
                        y1={laneY(Math.min(...item.lanes))}
                        y2={laneY(Math.max(...item.lanes))}
                        stroke={CATEGORIES[item.rel.category].color}
                        strokeDasharray="3 4"
                        strokeWidth={1.5}
                      />
                      {item.lanes.map((lane) => (
                        <rect
                          key={lane}
                          x={cx - 5}
                          y={laneY(lane) - 5}
                          width={10}
                          height={10}
                          transform={`rotate(45 ${cx} ${laneY(lane)})`}
                          fill={CATEGORIES[item.rel.category].color}
                        />
                      ))}
                    </g>
                  );
              }
            })}
          </svg>
        </g>
        <g>
          <text x={16} y={AXIS_HEIGHT + 22} className="fill-violet-300 text-[11px] font-semibold tracking-wider uppercase">
            Events
          </text>
          {lanes.map((lane, i) => {
            const gender = state.people[lane.personId]?.gender ?? "unknown";
            return (
              <g key={lane.personId} transform={`translate(16,${laneY(i)})`}>
                <circle r={12} cx={12} fill={GENDER_COLORS[gender][1]} />
                <text x={12} dy="0.35em" textAnchor="middle" className="fill-white text-[9px] font-bold">
                  {initials(lane.name)}
                </text>
                <text x={32} dy="0.35em" className="fill-foreground text-xs font-medium">
                  {lane.name.length > 16 ? `${lane.name.slice(0, 15)}…` : lane.name}
                </text>
              </g>
            );
          })}
          {lanes.length === 0 && items.length === 0 && (
            <text x={width / 2} y={lanesTop + 30} textAnchor="middle" className="fill-muted-foreground text-sm">
              No dated events or facts yet
            </text>
          )}
        </g>
        <defs>
          <linearGradient id="event-fill" x1="0" x2="1">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.75} />
            <stop offset="100%" stopColor="#ec4899" stopOpacity={0.55} />
          </linearGradient>
        </defs>
      </svg>

      <div className="sticky bottom-0 border-t border-border bg-background/85 backdrop-blur">
        <svg width={width} height={BRUSH_HEIGHT} className="block" data-testid="timeline-brush">
          <g transform={`translate(${LABEL_WIDTH},0)`}>
            {items.map((item) =>
              item.kind === "event" ? (
                <rect
                  key={item.key}
                  x={base(item.range.start)}
                  y={8 + ((rows.get(item.key) ?? 0) % 3) * 6}
                  width={Math.max(base(item.range.end) - base(item.range.start), 3)}
                  height={4}
                  rx={2}
                  className="fill-violet-400/70"
                />
              ) : (
                <circle key={item.key} cx={base(mid(item.range))} cy={32} r={2} fill={item.kind === "fact" ? KIND_COLORS[item.fact.kind] : "#94a3b8"} />
              ),
            )}
            <g ref={brushRef} className="[&_.selection]:fill-violet-400/15 [&_.selection]:stroke-violet-300/60" />
          </g>
          <text x={16} y={26} className="fill-muted-foreground text-[10px] tracking-wider uppercase">
            Overview
          </text>
        </svg>
      </div>

      {hover && <HoverCardBox hover={hover} state={state} />}
    </div>
  );
}

function Legend() {
  return (
    <div className="hidden items-center gap-3 rounded-md border border-border bg-card/80 px-3 py-1.5 text-[11px] text-muted-foreground backdrop-blur lg:flex">
      {Object.entries(KIND_COLORS).map(([kind, color]) => (
        <span key={kind} className="flex items-center gap-1">
          <span className="size-2 rounded-full" style={{ background: color }} />
          {kind}
        </span>
      ))}
      <span className="flex items-center gap-1">
        <span className="size-2 rotate-45 bg-emerald-400" />
        met
      </span>
    </div>
  );
}

interface HoverCardBoxProps {
  hover: Hover;
  state: RunState;
}

function HoverCardBox({ hover, state }: HoverCardBoxProps) {
  const { item } = hover;
  return (
    <div
      className="pointer-events-none absolute z-20 w-72 rounded-lg border border-border bg-popover/95 p-3 text-sm shadow-xl backdrop-blur"
      style={{ left: Math.min(hover.x + 14, window.innerWidth - 300), top: hover.y + 14 }}
    >
      {item.kind === "event" && (
        <>
          <div className="font-semibold">{item.event.name}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {item.event.when_text ?? item.event.start_date}
            {item.event.place ? ` · ${item.event.place}` : ""}
          </div>
          <div className="mt-2 text-xs">{item.event.people.map((p) => personName(state, p.person_id)).join(", ")}</div>
        </>
      )}
      {item.kind === "fact" && (
        <>
          <div className="flex items-center gap-2">
            <Badge variant="outline" style={{ color: KIND_COLORS[item.fact.kind], borderColor: KIND_COLORS[item.fact.kind] }}>
              {item.fact.kind}
            </Badge>
            {item.fact.disputed && <Badge variant="destructive">disputed</Badge>}
          </div>
          <div className="mt-2">{item.fact.text}</div>
          <div className="mt-1 text-xs text-muted-foreground">
            {item.fact.when_text ?? item.fact.start_date} · told by {personName(state, item.fact.told_by_person_id)}
          </div>
        </>
      )}
      {item.kind === "met" && (
        <>
          <div className="font-semibold">
            {personName(state, item.rel.a)} and {personName(state, item.rel.b)} met
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            {item.rel.met_on} · {item.rel.label}
          </div>
        </>
      )}
    </div>
  );
}

interface StoryFeedProps {
  state: RunState;
  items: Item[];
  recent: Set<string>;
  focusKey: string | null;
  onFocus: (key: string) => void;
}

// The events in date order, each with its story and facts. Undated facts are listed last, so dates
// that the model missed are easy to spot.
function StoryFeed({ state, items, recent, focusKey, onFocus }: StoryFeedProps) {
  const events = items.filter((i): i is Extract<Item, { kind: "event" }> => i.kind === "event");
  const facts = liveFacts(state);
  const undated = facts.filter((f) => !factRange(f, state) && f.kind !== "personal");

  return (
    <ScrollArea className="w-[360px] shrink-0 border-l border-border bg-card/40">
      <div className="p-4">
        <h3 className="mb-4 flex items-center gap-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
          <CalendarDays className="size-4" /> Stories
        </h3>
        <ol className="relative ml-2 border-l border-border">
          {events.map((item) => {
            const story = facts.find((f) => f.event_id === item.event.id && f.kind === "story" && f.detail);
            const own = facts.filter((f) => f.event_id === item.event.id && f !== story);
            const lit = recent.has(item.key) || focusKey === item.key;
            return (
              <li key={item.key} className="mb-5 ml-5" data-testid="story">
                <span
                  className={`absolute -left-[7px] mt-1.5 size-3.5 rounded-full border-2 border-background ${lit ? "bg-amber-300 shadow-[0_0_10px_rgb(252,211,77)]" : "bg-violet-400"}`}
                />
                <button className="w-full text-left" onClick={() => onFocus(item.key)}>
                  <Badge variant="secondary" className="mb-1 font-mono text-[10px]">
                    {item.event.start_date}
                    {item.event.when_text && item.event.when_text !== item.event.start_date ? ` · ${item.event.when_text}` : ""}
                  </Badge>
                  <div className={`rounded-lg border p-3 transition ${lit ? "border-amber-300/60 bg-amber-300/5" : "border-border bg-card hover:bg-accent/40"}`}>
                    <div className="font-semibold">{item.event.name}</div>
                    {item.event.place && (
                      <div className="mt-0.5 flex items-center gap-1 text-xs text-muted-foreground">
                        <MapPin className="size-3" />
                        {item.event.place}
                      </div>
                    )}
                    {story && <p className="mt-2 text-sm text-muted-foreground">{story.detail}</p>}
                    <ul className="mt-2 space-y-1">
                      {own.map((f) => (
                        <li key={f.id} className="flex gap-2 text-xs">
                          <span className="mt-1 size-1.5 shrink-0 rounded-full" style={{ background: KIND_COLORS[f.kind] }} />
                          <span className={f.disputed ? "text-red-300" : ""}>
                            {f.text}
                            {f.disputed && <TriangleAlert className="ml-1 inline size-3" />}
                          </span>
                        </li>
                      ))}
                    </ul>
                    <div className="mt-2 flex -space-x-1.5">
                      {item.event.people.map((p) => (
                        <span
                          key={p.person_id}
                          title={personName(state, p.person_id)}
                          className="flex size-6 items-center justify-center rounded-full border-2 border-card text-[9px] font-bold text-white"
                          style={{ background: GENDER_COLORS[state.people[p.person_id]?.gender ?? "unknown"][1] }}
                        >
                          {initials(personName(state, p.person_id))}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              </li>
            );
          })}
          {events.length === 0 && <li className="ml-5 text-sm text-muted-foreground">No dated events yet</li>}
        </ol>
        {undated.length > 0 && (
          <>
            <h3 className="mt-6 mb-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">
              Needs a date ({undated.length})
            </h3>
            <ul className="space-y-1.5">
              {undated.map((f) => (
                <li key={f.id} className="rounded-md border border-dashed border-border px-2.5 py-1.5 text-xs text-muted-foreground">
                  {f.text}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </ScrollArea>
  );
}
