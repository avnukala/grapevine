import OpenAI from "openai";
import type { GraphOp } from "../src/graph/ops.ts";
import { SYSTEM_PROMPT, OPS_SCHEMA, buildUserMessage } from "./prompt.ts";

// OPENAI_BASE_URL lets this point at any OpenAI-compatible endpoint: the
// hosted OpenAI API (default), a deployed gateway, or a local server like
// Ollama (http://localhost:11434/v1) or LM Studio (http://localhost:1234/v1).
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY ?? "not-needed",
  baseURL: process.env.OPENAI_BASE_URL,
});

const MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";

// Log every model request/response. Set GRAPEVINE_LOG=off to silence,
// or GRAPEVINE_LOG=verbose to also dump the full graph sent in each request.
const LOG = process.env.GRAPEVINE_LOG ?? "on";

// Calls the model with the current graph + new text and returns the proposed ops.
// Structured outputs guarantees the response is valid JSON matching OPS_SCHEMA.
export async function extractOps(
  graphJson: string,
  newText: string,
  context = "",
): Promise<GraphOp[]> {
  if (LOG !== "off") {
    console.log(`\n── [${new Date().toISOString()}] extract request → ${MODEL}`);
    console.log(`   text: ${JSON.stringify(newText)}`);
    if (LOG === "verbose") console.log(`   graph sent:\n${graphJson}`);
  }

  const response = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 2048,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: buildUserMessage(graphJson, newText, context) },
    ],
    response_format: {
      type: "json_schema",
      json_schema: { name: "graph_operations", schema: OPS_SCHEMA, strict: true },
    },
  });

  const rawText = response.choices[0]?.message?.content ?? "";

  if (LOG !== "off") {
    const u = response.usage;
    console.log(`   response (in ${u?.prompt_tokens ?? "?"} / out ${u?.completion_tokens ?? "?"} tok):`);
    console.log(`   ${rawText || "(no text content)"}`);
  }

  if (!rawText) return [];
  const parsed = JSON.parse(rawText) as { operations?: GraphOp[] };
  return parsed.operations ?? [];
}
