import { TriangleAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { CATEGORIES, GENDER_COLORS, KIND_COLORS, initials } from "../state/categories";
import { canonicalId, liveFacts, personName } from "../state/derive";
import type { RunState } from "../state/types";

interface PersonDetailsProps {
  state: RunState;
  personId: number | null;
  onClose: () => void;
}

export function PersonDetails({ state, personId, onClose }: PersonDetailsProps) {
  const person = personId !== null ? state.people[personId] : undefined;
  const same = (id: number) => canonicalId(state, id) === personId;
  const relationships = person ? Object.values(state.relationships).filter((r) => same(r.a) || same(r.b)) : [];
  const facts = person ? liveFacts(state).filter((f) => f.people.some((p) => same(p.person_id) && p.role !== "teller")) : [];
  const groups = person
    ? [...new Set(Object.values(state.members).filter((m) => same(m.person_id)).map((m) => state.groups[m.group_id]?.name))].filter(Boolean)
    : [];

  return (
    <Sheet open={person !== undefined} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-[420px] sm:max-w-[420px]" data-testid="person-details">
        {person && (
          <>
            <SheetHeader>
              <div className="flex items-center gap-3">
                <span
                  className="flex size-12 items-center justify-center rounded-full text-base font-bold text-white shadow-lg"
                  style={{ background: `linear-gradient(135deg, ${GENDER_COLORS[person.gender].join(", ")})` }}
                >
                  {initials(person.name)}
                </span>
                <div>
                  <SheetTitle className="text-xl">{person.name}</SheetTitle>
                  <SheetDescription>
                    {person.aliases.length > 0 ? `Also called ${person.aliases.join(", ")}` : "No aliases"}
                  </SheetDescription>
                </div>
              </div>
              {groups.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-2">
                  {groups.map((g) => (
                    <Badge key={g} variant="secondary">
                      {g}
                    </Badge>
                  ))}
                </div>
              )}
            </SheetHeader>
            <ScrollArea className="min-h-0 flex-1 px-4 pb-6">
              <h3 className="mb-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Relationships</h3>
              <ul className="space-y-2">
                {relationships.map((r) => (
                  <li key={r.id} className="rounded-lg border border-border bg-card p-3">
                    <div className="flex items-center gap-2">
                      <span className="size-2.5 rounded-full" style={{ background: CATEGORIES[r.category].color }} />
                      <span className="font-medium">{personName(state, same(r.a) ? r.b : r.a)}</span>
                      <Badge variant="outline" className="ml-auto">
                        {r.label}
                      </Badge>
                      {r.status === "past" && <Badge variant="secondary">past</Badge>}
                    </div>
                    {r.summary && <p className="mt-1.5 text-sm text-muted-foreground">{r.summary}</p>}
                    {r.met_on && <p className="mt-1 text-xs text-muted-foreground">Met {r.met_on}</p>}
                  </li>
                ))}
                {relationships.length === 0 && <li className="text-sm text-muted-foreground">None yet</li>}
              </ul>
              <Separator className="my-4" />
              <h3 className="mb-2 text-xs font-semibold tracking-wider text-muted-foreground uppercase">Facts</h3>
              <ul className="space-y-2">
                {facts.map((f) => (
                  <li key={f.id} className="flex gap-2 text-sm">
                    <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: KIND_COLORS[f.kind] }} />
                    <div>
                      <span className={f.disputed ? "text-red-700 dark:text-red-300" : ""}>{f.text}</span>
                      {f.disputed && <TriangleAlert className="ml-1 inline size-3.5 text-red-700 dark:text-red-300" />}
                      {(f.when_text || f.start_date) && (
                        <div className="text-xs text-muted-foreground">{f.when_text ?? f.start_date}</div>
                      )}
                      {f.detail && <p className="mt-1 text-xs text-muted-foreground">{f.detail}</p>}
                    </div>
                  </li>
                ))}
                {facts.length === 0 && <li className="text-sm text-muted-foreground">None yet</li>}
              </ul>
            </ScrollArea>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
