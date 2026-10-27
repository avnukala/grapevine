import { CATEGORIES } from "./categories";
import type { DatePrecision, Fact, Person, Relationship, RunState } from "./types";

// Values the views compute from the records. None of them is stored.

export function canonicalId(state: RunState, id: number): number {
  let current = id;
  for (let person = state.people[current]; person?.merged_into_id; person = state.people[current]) {
    current = person.merged_into_id;
  }
  return current;
}

export function visiblePeople(state: RunState): Person[] {
  return Object.values(state.people).filter((p) => p.merged_into_id === null);
}

// Everyone on the roster, and anyone a voice belongs to. Mirrors `Graph.room` in graph.py.
export function roomIds(state: RunState): number[] {
  const ids = Object.values(state.speakers).filter((id): id is number => id !== null);
  return [...new Set([...state.roster, ...ids])];
}

export function liveFacts(state: RunState): Fact[] {
  return Object.values(state.facts).filter((f) => f.revised_by_fact_id === null);
}

export function personName(state: RunState, id: number | null | undefined): string {
  if (id === null || id === undefined) return "Unknown speaker";
  return state.people[canonicalId(state, id)]?.name ?? `Person ${id}`;
}

// The node label is the alias people use most that no one else on screen shares, for example "Gao"
// when two Andrews exist. Use counts the words of the transcript.
export function nodeLabels(state: RunState): Record<number, string> {
  const people = visiblePeople(state);
  const owners = new Map<string, number>();
  for (const person of people) {
    for (const name of candidates(person)) {
      const key = name.toLowerCase();
      owners.set(key, owners.has(key) && owners.get(key) !== person.id ? -1 : person.id);
    }
  }
  const transcript = Object.values(state.segments)
    .map((s) => s.text.toLowerCase())
    .join(" ");
  const labels: Record<number, string> = {};
  for (const person of people) {
    const unique = candidates(person).filter((n) => owners.get(n.toLowerCase()) === person.id);
    const first = person.name.split(" ")[0];
    // The transcript keeps the misheard spelling of a renamed person, which still counts as their first name.
    const heard = [first, ...person.aliases.map((a) => a.split(" ")[0]).filter((w) => misheard(w, first))];
    const count = (name: string) => transcript.split(name.toLowerCase()).length - 1;
    const used = (name: string) => (name === first ? heard : [name]).reduce((sum, n) => sum + count(n), 0);
    const best = [...unique].sort(
      (x, y) => used(y) - used(x) || Number(x !== first) - Number(y !== first) || x.length - y.length,
    )[0];
    labels[person.id] = best ?? person.name;
  }
  return labels;
}

function candidates(person: Person): string[] {
  const words = person.name.split(" ");
  const aliases = person.aliases.filter((a) => !misheard(a.split(" ")[0], words[0]));
  return [...new Set([words[0], ...aliases, ...(words.length > 1 ? [words[words.length - 1]] : []), person.name])];
}

// An old spelling of the first name, such as "Kiri" for Keri or "Sophiely" for Sophie, stays as an alias so the
// same mishearing still finds the person. It is not a name people use, so it never labels them.
function misheard(word: string, first: string): boolean {
  const [a, b] = [word.toLowerCase(), first.toLowerCase()];
  if (a === b) return false;
  if (a === `${b}ly`) return true;
  if (Math.abs(a.length - b.length) > 1) return false;
  let i = 0;
  while (i < a.length && a[i] === b[i]) i++;
  return a.slice(i + 1) === b.slice(i + 1) || a.slice(i) === b.slice(i + 1) || a.slice(i + 1) === b.slice(i);
}

export interface WeightFactors {
  yearsKnown: number;
  recency: number;
  closeness: number;
  sentiment: number;
  category: number;
}

export const DEFAULT_FACTORS: WeightFactors = {
  yearsKnown: 0.6,
  recency: 0.15,
  closeness: 0.15,
  sentiment: 0.05,
  category: 0.05,
};

const FULL_YEARS = 15;

// Edge weight from 0 to 1, a weighted sum of the raw inputs of a relationship. A missing input counts as
// the middle of its range.
export function edgeWeight(rel: Relationship, factors: WeightFactors, today = new Date()): number {
  const met = rel.met_on ? parseDate(rel.met_on, "year") : null;
  const years = met ? (today.getTime() - met.start.getTime()) / (365.25 * 24 * 3600 * 1000) : null;
  const parts: [number, number][] = [
    [factors.yearsKnown, years === null ? 0.3 : Math.min(Math.max(years, 0), FULL_YEARS) / FULL_YEARS],
    [factors.recency, rel.recency === null ? 0.5 : (rel.recency - 1) / 4],
    [factors.closeness, rel.closeness === null ? 0.5 : (rel.closeness - 1) / 4],
    [factors.sentiment, rel.sentiment === null ? 0.5 : (rel.sentiment + 2) / 4],
    [factors.category, CATEGORIES[rel.category].weight * (rel.status === "past" ? 0.5 : 1)],
  ];
  const total = parts.reduce((sum, [w]) => sum + w, 0);
  if (total === 0) return 0.5;
  return parts.reduce((sum, [w, value]) => sum + w * value, 0) / total;
}

export interface DateRange {
  start: Date;
  end: Date;
}

const SEASON_MONTHS = 3;

// Turns "2016", "2016-09", or "2016-09-14" into the span of time it names.
export function parseDate(value: string, precision: DatePrecision | null): DateRange | null {
  const match = /^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$/.exec(value);
  if (!match) return null;
  const year = Number(match[1]);
  const month = match[2] ? Number(match[2]) - 1 : null;
  const day = match[3] ? Number(match[3]) : null;
  if (month === null) return { start: new Date(year, 0, 1), end: new Date(year + 1, 0, 1) };
  if (day === null) {
    const months = precision === "season" ? SEASON_MONTHS : 1;
    return { start: new Date(year, month, 1), end: new Date(year, month + months, 1) };
  }
  return { start: new Date(year, month, day), end: new Date(year, month, day + 1) };
}

export function dateRange(
  start: string | null,
  end: string | null,
  precision: DatePrecision | null,
): DateRange | null {
  const from = start ? parseDate(start, precision) : null;
  if (!from) return null;
  const to = end ? parseDate(end, precision) : null;
  return { start: from.start, end: to && to.end > from.start ? to.end : from.end };
}
