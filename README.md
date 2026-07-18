# grapevine

Real-time relationship graph from conversation.

Step 1 of the build: a synchronous pipeline that turns **text → graph operations → a rendered social graph**. Nodes are people; edges are relationships (romantic, family, friend, professional, …), each with a specific label like "childhood friend" or "girlfriend". Later steps add the interval loop and live speech input on top of this same pipeline.

## Architecture

```
textarea → POST /api/extract (server + Claude) → GraphOp[]
        → applyOps(currentGraph, ops) → newGraph → Cytoscape render
```

- **The client owns the graph state** (`src/graph`). The LLM only ever *proposes* diffs.
- **The server** (`server/`) holds the Anthropic API key and calls Claude with structured outputs so the response is always a validated list of operations.
- **Model:** `claude-haiku-4-5` by default (change with `GRAPEVINE_MODEL`).

### Key files

| File | Role |
| --- | --- |
| `src/graph/types.ts` | `Person`, `Relationship`, `SocialGraph`, relationship categories |
| `src/graph/relationshipTypes.ts` | Edge categories → color, default directedness, examples (single source of truth) |
| `src/graph/ops.ts` | `GraphOp` discriminated union (the LLM's output contract) |
| `src/graph/reducer.ts` | `applyOps` — pure, idempotent, owns dedup + cleanup |
| `server/prompt.ts` | System prompt + JSON schema for structured outputs |
| `server/extract.ts` | The Claude call |
| `src/ui/GraphView.tsx` | Cytoscape renderer (`SocialGraph` → elements) |

## Setup

```sh
npm install
cp .env.example .env      # add your ANTHROPIC_API_KEY
npm run dev               # server on :8787, web on :5173
```

Open http://localhost:5173, type a description of some people and how they know each other, and press **⌘/Ctrl+Enter**.

## Design notes

- **One relationship per pair of people.** Re-describing a pair updates the same edge, so "dating → married" is a correction, not a duplicate.
- **Dedup** is driven by passing the existing people (with aliases) into every prompt so the model reuses ids instead of creating "Sarah" and "Sarah Chen" as separate nodes.
- **Corrections** are first-class: the ops include `remove_*` and `update_*`, not just adds.
