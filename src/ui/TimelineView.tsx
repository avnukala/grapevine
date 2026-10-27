import * as d3 from "d3";
import { CalendarDays, MapPin, TriangleAlert } from "lucide-react";
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { CATEGORIES, GENDER_COLORS, KIND_COLORS, initials } from "../state/categories";
import { canonicalId, dateRange, liveFacts, personName, roomIds } from "../state/derive";
import type { DateRange } from "../state/derive";
import type { Fact, GraphEvent, PersonRole, Relationship, RunState } from "../state/types";

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
// The legend sits right of the view switch, so it shows only when the plot has room for both.
const LEGEND_MIN_PLOT = 500;
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

interface EventLabel {
  x: number;
  anchor: "start" | "end";
  start: number;
  end: number;
}

// Puts an event label inside the visible part of a wide bar, else after the bar, or before it when it would run off the chart.
function eventLabel(item: Extract<Item, { kind: "event" }>, x: d3.ScaleTime<number, number>, plotWidth: number): EventLabel {
  const x0 = x(item.range.start);
  const x1 = x0 + Math.max(x(item.range.end) - x0, 8);
  const text = item.event.name.length * 6.5;
  const left = Math.max(x0, 0);
  if (Math.min(x1, plotWidth) - left > text + 16) return { x: left + 8, anchor: "start", start: x0, end: x1 };
  if (x1 + 6 + text <= plotWidth) return { x: x1 + 6, anchor: "start", start: x0, end: x1 + 6 + text };
  return { x: x0 - 6, anchor: "end", start: x0 - 6 - text, end: x1 };
}

// Packs event bars into rows so that bars and their labels do not overlap at the given scale.
function packEvents(items: Item[], x: d3.ScaleTime<number, number>, plotWidth: number): Map<string, number> {
  const rowEnds: number[] = [];
  const rows = new Map<string, number>();
  for (const item of items) {
    if (item.kind !== "event") continue;
    const { start, end } = eventLabel(item, x, plotWidth);
    let row = rowEnds.findIndex((rowEnd) => rowEnd + 20 < start);
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

function TimelineView({ state, recentFactIds, recentEventIds }: TimelineViewProps) {
  const { lanes, items } = useMemo(() => layout(state), [state]);
  const [focusKey, setFocusKey] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const recent = new Set([...recentFactIds.map((id) => `f${id}`), ...recentEventIds.map((id) => `e${id}`)]);
  const newest = [...items].reverse().find((item) => recent.has(item.key))?.key ?? null;

  useEffect(() => {
    if (newest) setFocusKey(newest);
  }, [newest]);

  return (
    <div className="@container flex size-full min-h-0" data-testid="timeline">
      <SwimlaneChart
        state={state}
        lanes={lanes}
        items={items}
        recent={recent}
        focusKey={focusKey}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
      />
      <StoryFeed state={state} items={items} recent={recent} focusKey={focusKey} onFocus={setFocusKey} />
      <ItemDetails state={state} item={items.find((i) => i.key === selectedKey) ?? null} onClose={() => setSelectedKey(null)} />
    </div>
  );
}

interface SwimlaneChartProps {
  state: RunState;
  lanes: Lane[];
  items: Item[];
  recent: Set<string>;
  focusKey: string | null;
  selectedKey: string | null;
  onSelect: (key: string) => void;
}

// The pointer and the visible part of the chart above the overview, both in the coordinates of the scrolled content.
interface Hover {
  x: number;
  y: number;
  view: { left: number; top: number; right: number; bottom: number };
  item: Item;
}

function SwimlaneChart({ state, lanes, items, recent, focusKey, selectedKey, onSelect }: SwimlaneChartProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<SVGSVGElement>(null);
  const brushRef = useRef<SVGGElement>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const [width, setWidth] = useState(900);
  const [transform, setTransform] = useState(d3.zoomIdentity);
  const [hover, setHover] = useState<Hover | null>(null);
  const [keyboardKey, setKeyboardKey] = useState<string | null>(null);

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
  const rows = useMemo(() => packEvents(items, base, plotWidth), [items, base, plotWidth]);
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
  // The card is absolute inside the scrolled box, so the scroll offset turns viewport offsets into content offsets.
  const show = (event: React.MouseEvent, item: Item) => {
    const wrap = wrapRef.current!;
    const box = wrap.getBoundingClientRect();
    const left = wrap.scrollLeft;
    const top = wrap.scrollTop;
    setHover({
      x: event.clientX - box.left + left,
      y: event.clientY - box.top + top,
      view: { left, top, right: left + wrap.clientWidth, bottom: top + wrap.clientHeight - BRUSH_HEIGHT },
      item,
    });
  };
  const lit = (key: string) => recent.has(key) || focusKey === key || selectedKey === key || hover?.item.key === key;
  const press = (item: Item, label: string) => ({
    tabIndex: 0,
    role: "button",
    "aria-label": label,
    "aria-pressed": selectedKey === item.key,
    onMouseMove: (e: React.MouseEvent) => show(e, item),
    onMouseLeave: () => setHover(null),
    onFocus: () => setKeyboardKey(item.key),
    onBlur: () => setKeyboardKey(null),
    onClick: () => onSelect(item.key),
    onKeyDown: (e: React.KeyboardEvent) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      onSelect(item.key);
    },
  });

  return (
    <div ref={wrapRef} className="relative min-w-0 flex-1 overflow-auto">
      <div className="sticky top-0 z-10 flex h-14 items-center justify-end gap-2 bg-background px-4">
        {width - LABEL_WIDTH >= LEGEND_MIN_PLOT && <Legend />}
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
              className={i % 2 ? "fill-foreground/[0.015]" : "fill-foreground/[0.035]"}
            />
          ))}
          <rect x={-LABEL_WIDTH} y={AXIS_HEIGHT} width={width} height={eventsHeight} className="fill-primary/[0.05]" />
        </g>
        <g transform={`translate(${LABEL_WIDTH},0)`}>
          <svg ref={chartRef} width={plotWidth} height={height} className="cursor-grab overflow-hidden" data-testid="timeline-chart">
            <rect width={plotWidth} height={height} fill="transparent" />
            {ticks.map((tick) => (
              <g key={tick.getTime()} transform={`translate(${x(tick)},0)`}>
                <line y1={AXIS_HEIGHT - 4} y2={height} className="stroke-foreground/[0.08]" />
                {Math.min(x(tick), plotWidth - x(tick)) > format(tick).length * 3.5 && (
                  <text y={AXIS_HEIGHT - 12} textAnchor="middle" className="fill-muted-foreground text-[11px]">
                    {format(tick)}
                  </text>
                )}
              </g>
            ))}
            {items.map((item) => {
              const on = lit(item.key);
              const ringed = selectedKey === item.key || keyboardKey === item.key;
              const cx = x(mid(item.range));
              switch (item.kind) {
                case "event": {
                  const x0 = x(item.range.start);
                  const w = Math.max(x(item.range.end) - x0, 8);
                  const y = AXIS_HEIGHT + 8 + (rows.get(item.key) ?? 0) * EVENT_ROW;
                  const label = eventLabel(item, x, plotWidth);
                  return (
                    <g key={item.key} className="cursor-pointer" {...press(item, item.event.name)} data-testid="timeline-event">
                      <rect
                        x={x0}
                        y={y}
                        width={w}
                        height={EVENT_ROW - 6}
                        rx={6}
                        fill="url(#event-fill)"
                        strokeWidth={ringed ? 2.5 : 1}
                        className={ringed ? "stroke-foreground" : on ? "stroke-warn" : "stroke-primary/50"}
                      />
                      {x0 + w >= 0 && x0 <= plotWidth && (
                        <text
                          x={label.x}
                          y={y + 14}
                          textAnchor={label.anchor}
                          className="fill-foreground text-[11px] font-semibold"
                        >
                          {item.event.name}
                        </text>
                      )}
                    </g>
                  );
                }
                case "fact":
                  return (
                    <g key={item.key} className="cursor-pointer" {...press(item, item.fact.text)} data-testid="timeline-fact">
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
                          r={on ? 8 : 6}
                          fill={KIND_COLORS[item.fact.kind]}
                          stroke={ringed ? "var(--foreground)" : item.fact.disputed ? "var(--alert)" : "var(--background)"}
                          strokeWidth={item.fact.disputed || ringed ? 3 : 2}
                          className={on ? "" : ""}
                        />
                      ))}
                    </g>
                  );
                case "met":
                  return (
                    <g
                      key={item.key}
                      className="cursor-pointer"
                      {...press(item, `${personName(state, item.rel.a)} and ${personName(state, item.rel.b)} met`)}
                      data-testid="timeline-met"
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
                          x={cx - (on ? 7 : 5)}
                          y={laneY(lane) - (on ? 7 : 5)}
                          width={on ? 14 : 10}
                          height={on ? 14 : 10}
                          transform={`rotate(45 ${cx} ${laneY(lane)})`}
                          fill={CATEGORIES[item.rel.category].color}
                          stroke={ringed ? "var(--foreground)" : "none"}
                          strokeWidth={2}
                        />
                      ))}
                    </g>
                  );
              }
            })}
          </svg>
        </g>
        <g>
          <text x={16} y={AXIS_HEIGHT + 22} className="fill-primary text-[11px] font-semibold tracking-wider uppercase">
            Events
          </text>
          {lanes.map((lane, i) => {
            const gender = state.people[lane.personId]?.gender ?? "unknown";
            return (
              <g key={lane.personId} transform={`translate(16,${laneY(i)})`}>
                <circle r={12} cx={12} fill={GENDER_COLORS[gender]} />
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
            <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.6} />
            <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.6} />
          </linearGradient>
        </defs>
      </svg>

      <div className="sticky bottom-0 border-t border-border bg-background">
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
                  className="fill-primary/70"
                />
              ) : (
                <circle key={item.key} cx={base(mid(item.range))} cy={32} r={2} fill={item.kind === "fact" ? KIND_COLORS[item.fact.kind] : "var(--muted-foreground)"} />
              ),
            )}
            <g ref={brushRef} className="[&_.selection]:fill-primary/15 [&_.selection]:stroke-primary/60" />
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
    <div className="flex items-center gap-3 rounded-md border border-border bg-card px-3 py-1.5 text-[11px] text-muted-foreground">
      {Object.entries(KIND_COLORS).map(([kind, color]) => (
        <span key={kind} className="flex items-center gap-1">
          <span className="size-2 rounded-full" style={{ background: color }} />
          {kind}
        </span>
      ))}
      <span className="flex items-center gap-1">
        <span className="size-2 rotate-45 bg-speak" />
        met
      </span>
    </div>
  );
}

interface HoverCardBoxProps {
  hover: Hover;
  state: RunState;
}

const HOVER_GAP = 14;
const HOVER_MARGIN = 8;

// Puts the card below and right of the pointer, and flips it to the other side when it would leave the visible chart.
function HoverCardBox({ hover, state }: HoverCardBoxProps) {
  const { item, view } = hover;
  const ref = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const card = ref.current!;
    const fitsRight = hover.x + HOVER_GAP + card.offsetWidth <= view.right - HOVER_MARGIN;
    const fitsBelow = hover.y + HOVER_GAP + card.offsetHeight <= view.bottom - HOVER_MARGIN;
    const left = fitsRight ? hover.x + HOVER_GAP : hover.x - HOVER_GAP - card.offsetWidth;
    const top = fitsBelow ? hover.y + HOVER_GAP : hover.y - HOVER_GAP - card.offsetHeight;
    card.style.left = `${Math.max(view.left + HOVER_MARGIN, left)}px`;
    card.style.top = `${Math.max(view.top + HOVER_MARGIN, top)}px`;
  });

  return (
    <div
      ref={ref}
      className="pointer-events-none absolute z-20 w-72 rounded-lg border border-border bg-popover p-3 text-sm"
      data-testid="timeline-hover"
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

function datesText(when: string | null, start: string | null, end: string | null): string {
  const dates = start && end && end !== start ? `${start} to ${end}` : start;
  if (!when) return dates ?? "No date";
  return dates && !when.includes(dates) ? `${when} · ${dates}` : when;
}

interface PeopleListProps {
  state: RunState;
  people: PersonRole[];
}

// Tellers are left out, as the timeline lanes leave them out.
function PeopleList({ state, people }: PeopleListProps) {
  const ids = [...new Set(people.filter((p) => p.role !== "teller").map((p) => canonicalId(state, p.person_id)))];
  return (
    <ul className="space-y-1.5">
      {ids.map((id) => (
        <li key={id} className="flex items-center gap-2">
          <span
            className="flex size-6 items-center justify-center rounded-full text-[9px] font-bold text-ink"
            style={{ background: GENDER_COLORS[state.people[id]?.gender ?? "unknown"] }}
          >
            {initials(personName(state, id))}
          </span>
          {personName(state, id)}
        </li>
      ))}
      {ids.length === 0 && <li className="text-muted-foreground">Nobody named</li>}
    </ul>
  );
}

interface HeadingProps {
  children: React.ReactNode;
}

function Heading({ children }: HeadingProps) {
  return <h3 className="mt-4 mb-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase first:mt-0">{children}</h3>;
}

interface ItemDetailsProps {
  state: RunState;
  item: Item | null;
  onClose: () => void;
}

// Not modal, so a click on another dot swaps the content while the sheet stays open.
function ItemDetails({ state, item, onClose }: ItemDetailsProps) {
  return (
    <Sheet open={item !== null} onOpenChange={(open) => !open && onClose()} modal={false}>
      <SheetContent
        className="w-[420px] sm:max-w-[420px]"
        data-testid="timeline-details"
        onInteractOutside={(e) => (e.target as Element).closest("[data-testid=timeline-chart]") && e.preventDefault()}
      >
        {item?.kind === "fact" && <FactDetails state={state} fact={item.fact} />}
        {item?.kind === "event" && <EventDetails state={state} event={item.event} />}
        {item?.kind === "met" && <MetDetails state={state} rel={item.rel} />}
      </SheetContent>
    </Sheet>
  );
}

interface FactDetailsProps {
  state: RunState;
  fact: Fact;
}

function FactDetails({ state, fact }: FactDetailsProps) {
  const event = fact.event_id ? state.events[fact.event_id] : undefined;
  const earlier = Object.values(state.facts).filter((f) => f.revised_by_fact_id === fact.id);
  const segments = fact.segment_ids.map((id) => state.segments[id]).filter(Boolean);
  const dated = fact.start_date || fact.when_text || !event ? fact : event;
  return (
    <>
      <SheetHeader>
        <div className="flex items-center gap-2 pr-8">
          <Badge variant="outline" style={{ color: KIND_COLORS[fact.kind], borderColor: KIND_COLORS[fact.kind] }}>
            {fact.kind}
          </Badge>
          {fact.disputed && (
            <Badge variant="destructive">
              <TriangleAlert /> disputed
            </Badge>
          )}
          {earlier.length > 0 && <Badge variant="secondary">revised</Badge>}
        </div>
        <SheetTitle className="text-lg leading-snug">{fact.text}</SheetTitle>
        <SheetDescription>
          {datesText(dated.when_text, dated.start_date, dated.end_date)} · told by {personName(state, fact.told_by_person_id)}
        </SheetDescription>
      </SheetHeader>
      <ScrollArea className="min-h-0 flex-1 px-4 pb-6">
        {fact.detail && <p className="mb-4 text-sm">{fact.detail}</p>}
        <Heading>People</Heading>
        <PeopleList state={state} people={fact.people} />
        {event && (
          <>
            <Heading>Event</Heading>
            <div className="font-medium">{event.name}</div>
            <div className="text-xs text-muted-foreground">{datesText(event.when_text, event.start_date, event.end_date)}</div>
          </>
        )}
        {earlier.length > 0 && (
          <>
            <Heading>Earlier versions</Heading>
            <ul className="space-y-1.5">
              {earlier.map((f) => (
                <li key={f.id} className="text-muted-foreground line-through">
                  {f.text}
                </li>
              ))}
            </ul>
          </>
        )}
        {segments.length > 0 && (
          <>
            <Separator className="my-4" />
            <Heading>In their words</Heading>
            <ul className="space-y-2">
              {segments.map((s) => (
                <li key={s.id} className="rounded-lg border border-border bg-card p-3">
                  <div className="text-xs font-medium text-muted-foreground">{personName(state, s.person_id)}</div>
                  <p className="mt-1">{s.text}</p>
                </li>
              ))}
            </ul>
          </>
        )}
      </ScrollArea>
    </>
  );
}

interface EventDetailsProps {
  state: RunState;
  event: GraphEvent;
}

function EventDetails({ state, event }: EventDetailsProps) {
  const facts = liveFacts(state).filter((f) => f.event_id === event.id);
  return (
    <>
      <SheetHeader>
        <Badge variant="outline" className="text-primary">
          event
        </Badge>
        <SheetTitle className="pr-8 text-lg leading-snug">{event.name}</SheetTitle>
        <SheetDescription>
          {datesText(event.when_text, event.start_date, event.end_date)}
          {event.place ? ` · ${event.place}` : ""}
        </SheetDescription>
      </SheetHeader>
      <ScrollArea className="min-h-0 flex-1 px-4 pb-6">
        <Heading>People</Heading>
        <PeopleList state={state} people={event.people} />
        <Heading>Facts</Heading>
        <ul className="space-y-2">
          {facts.map((f) => (
            <li key={f.id} className="flex gap-2">
              <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: KIND_COLORS[f.kind] }} />
              <div>
                <span className={f.disputed ? "text-alert" : ""}>{f.text}</span>
                {f.disputed && <TriangleAlert className="ml-1 inline size-3.5 text-alert" />}
                {f.detail && <p className="mt-1 text-xs text-muted-foreground">{f.detail}</p>}
                <div className="text-xs text-muted-foreground">told by {personName(state, f.told_by_person_id)}</div>
              </div>
            </li>
          ))}
          {facts.length === 0 && <li className="text-muted-foreground">None yet</li>}
        </ul>
      </ScrollArea>
    </>
  );
}

interface MetDetailsProps {
  state: RunState;
  rel: Relationship;
}

function MetDetails({ state, rel }: MetDetailsProps) {
  return (
    <>
      <SheetHeader>
        <div className="flex items-center gap-2 pr-8">
          <Badge variant="outline" style={{ color: CATEGORIES[rel.category].color, borderColor: CATEGORIES[rel.category].color }}>
            {rel.label}
          </Badge>
          {rel.status === "past" && <Badge variant="secondary">past</Badge>}
        </div>
        <SheetTitle className="text-lg leading-snug">
          {personName(state, rel.a)} and {personName(state, rel.b)} met
        </SheetTitle>
        <SheetDescription>{rel.met_on}</SheetDescription>
      </SheetHeader>
      <ScrollArea className="min-h-0 flex-1 px-4 pb-6">
        {rel.summary && <p className="mb-4 text-sm">{rel.summary}</p>}
        <Heading>People</Heading>
        <PeopleList state={state} people={[rel.a, rel.b].map((person_id) => ({ person_id, role: "met" }))} />
      </ScrollArea>
    </>
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
    <ScrollArea className="hidden w-[360px] shrink-0 border-l border-border bg-card/40 @4xl:block">
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
                  className={`absolute -left-[7px] mt-1.5 size-3.5 rounded-full border-2 border-background ${lit ? "bg-warn" : "bg-primary"}`}
                />
                <button className="w-full text-left" onClick={() => onFocus(item.key)}>
                  <Badge variant="secondary" className="mb-1 font-mono text-[10px]">
                    {item.event.start_date}
                    {item.event.when_text && item.event.when_text !== item.event.start_date ? ` · ${item.event.when_text}` : ""}
                  </Badge>
                  <div className={`rounded-lg border p-3 transition ${lit ? "border-warn/60 bg-warn/5" : "border-border bg-card hover:bg-accent/40"}`}>
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
                          <span className={f.disputed ? "text-alert" : ""}>
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
                          className="flex size-6 items-center justify-center rounded-full border-2 border-card text-[9px] font-bold text-ink"
                          style={{ background: GENDER_COLORS[state.people[p.person_id]?.gender ?? "unknown"] }}
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

// Live runs stream a message every few hundred milliseconds. A view redraws only when its own props change.
const MemoizedTimelineView = memo(TimelineView);
export { MemoizedTimelineView as TimelineView };
