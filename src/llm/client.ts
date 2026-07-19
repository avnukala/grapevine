import type { GraphOp } from "../graph/ops";
import type { SocialGraph } from "../graph/types";
import { serializeGraphForPrompt } from "../graph/serialize";

// Ask the server to turn `text` into graph operations, given the current graph
// and a window of recently-processed text (for resolving references like "he"/"she"
// or an unnamed subject mentioned a moment ago).
export async function extractOps(
  graph: SocialGraph,
  text: string,
  context = "",
): Promise<GraphOp[]> {
  const res = await fetch("/api/extract", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ graph: serializeGraphForPrompt(graph), text, context }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`extract failed (${res.status}): ${detail}`);
  }
  const data = (await res.json()) as { operations?: GraphOp[] };
  return data.operations ?? [];
}
