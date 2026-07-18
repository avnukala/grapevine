import "dotenv/config";
import express from "express";
import { extractOps } from "./extract.ts";

const app = express();
app.use(express.json({ limit: "1mb" }));

// POST /api/extract  { graph: <serialized graph JSON string>, text: string }
//   -> { operations: GraphOp[] }
// The client owns the graph state; the server only turns text into proposed ops.
app.post("/api/extract", async (req, res) => {
  try {
    const { graph, text } = req.body ?? {};
    if (typeof text !== "string" || typeof graph !== "string") {
      return res.status(400).json({ error: "expected { graph: string, text: string }" });
    }
    if (!text.trim()) return res.json({ operations: [] });

    const operations = await extractOps(graph, text);
    res.json({ operations });
  } catch (err) {
    console.error("extract failed:", err);
    res.status(500).json({ error: "extraction failed", detail: String(err) });
  }
});

const port = Number(process.env.PORT ?? 8787);
app.listen(port, () => {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn("⚠  ANTHROPIC_API_KEY is not set — /api/extract will fail. Copy .env.example to .env.");
  }
  console.log(`grapevine server listening on http://localhost:${port}`);
});
