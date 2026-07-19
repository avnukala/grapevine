import Anthropic from "@anthropic-ai/sdk";
import type { GraphOp } from "../src/graph/ops.ts";
import { SYSTEM_PROMPT, OPS_SCHEMA, buildUserMessage } from "./prompt.ts";

const client = new Anthropic(); // reads ANTHROPIC_API_KEY from the environment

const MODEL = process.env.GRAPEVINE_MODEL ?? "claude-haiku-4-5";

// Log every Claude request/response. Set GRAPEVINE_LOG=off to silence,
// or GRAPEVINE_LOG=verbose to also dump the full graph sent in each request.
const LOG = process.env.GRAPEVINE_LOG ?? "on";

// Calls Claude with the current graph + new text and returns the proposed ops.
// Structured outputs guarantees the response is valid JSON matching OPS_SCHEMA.
export async function extractOps(
  graphJson: string,
  newText: string,
  context = "",
): Promise<GraphOp[]> {
  // `output_config` is the canonical structured-outputs param. Cast to any so we
  // don't depend on the installed SDK's typings including it yet.
  const params = {
    model: MODEL,
    // Generous cap: a rich input can produce many ops with long descriptions.
    // Too small a cap truncates the JSON mid-string and breaks JSON.parse.
    max_tokens: 8192,
    system: SYSTEM_PROMPT,
    output_config: {
      format: { type: "json_schema", schema: OPS_SCHEMA },
    },
    messages: [{ role: "user", content: buildUserMessage(graphJson, newText, context) }],
  };

  if (LOG !== "off") {
    console.log(`\n── [${new Date().toISOString()}] extract request → ${MODEL}`);
    console.log(`   text: ${JSON.stringify(newText)}`);
    if (LOG === "verbose") console.log(`   graph sent:\n${graphJson}`);
  }

  const response = await client.messages.create(
    params as unknown as Anthropic.MessageCreateParamsNonStreaming,
  );

  const textBlock = response.content.find((b) => b.type === "text");
  const rawText = textBlock && textBlock.type === "text" ? textBlock.text : "";

  if (LOG !== "off") {
    const u = response.usage;
    console.log(`   response (in ${u.input_tokens} / out ${u.output_tokens} tok, stop=${response.stop_reason}):`);
    console.log(`   ${rawText || "(no text block)"}`);
  }

  // If the model hit the token cap, the JSON is truncated (invalid). Fail with a
  // clear, actionable message instead of a raw JSON.parse crash.
  if (response.stop_reason === "max_tokens") {
    throw new Error(
      "The model's response was too long and got cut off (hit max_tokens). Try describing fewer people/relationships per turn.",
    );
  }

  if (!rawText) return [];
  try {
    const parsed = JSON.parse(rawText) as { operations?: GraphOp[] };
    return parsed.operations ?? [];
  } catch {
    throw new Error("The model returned malformed JSON (likely truncated). Try a shorter input.");
  }
}
