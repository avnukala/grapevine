# grapevine

Real-time relationship graph from conversation.

## Setup

```sh
npm install
cp .env.example .env      # creates your own .env. Must add your OPENAI_API_KEY (or point OPENAI_BASE_URL at a local model server)
npm run dev               # access at localhost:5173
```

Open http://localhost:5173, type a description of some people and how they know each other, and press **⌘/Ctrl+Enter**.