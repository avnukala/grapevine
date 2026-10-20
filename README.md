# Grapevine

Grapevine listens to a group conversation and builds a knowledge graph of the people in it. The graph shows who knows whom, how they met, the stories they tell, and the groups they belong to. A timeline shows when each relationship and story happened. Everything runs on one laptop, and no data goes to a hosted API.

## How it works

- A Python back end in `backend/` owns all server work: audio input, speaker detection, transcription, model calls, the graph, and SQLite.
- NVIDIA Nemotron 3 Diarization finds who speaks when, for up to 8 speakers. WavLM voice embeddings check each turn against the voices that the app saves during introductions.
- Whisper large-v3-turbo transcribes each speaker turn on Apple silicon through MLX.
- A local model in LM Studio extracts people, facts, events, groups, and relationships from each turn. A second worker writes questions for the prompt panel. At the end of a session, the model writes stories and relationship summaries.
- The back end writes each change to SQLite and sends it to the client over a WebSocket. The React client only shows views and sends user actions.
- The app stores each model call. A replay with the same input uses the stored response again, so an unchanged replay gives the same result.

## Setup

You need an Apple silicon Mac, Node 22, [uv](https://docs.astral.sh/uv/), and [LM Studio](https://lmstudio.ai).

1. Install the client and the back end:

   ```sh
   npm install
   cd backend && uv sync && cd ..
   ```

2. Load the extraction model in LM Studio and start its server. The first run of the back end downloads the speech models from Hugging Face.

   ```sh
   lms load google/gemma-4-e4b --identifier gemma-4-e4b --context-length 32768
   lms server start
   ```

3. If you need to change a setting, copy `.env.example` to `.env` and edit it.

## Run the app

```sh
npm run dev
```

Open http://localhost:5173. The API listens on port 8790.

- To run a live session, create a session and click **Start live**. Let each person say their name, then click **Done with introductions**.
- To replay a recording, pick a session with audio and click **Replay 1×** or **Replay fast**.
- To check a run, open **Debug and replay**. It plays the audio and shows the graph, the timeline, the transcript, the model calls, and the change log at the current audio time. Only dev builds have this screen.
- To build the global graph, accept one run for each session. Then click **Merge accepted runs** on the **Global graph** screen.

## Replay and score from the command line

```sh
cd backend
uv run grapevine replay ../data/samples/kickback-3p.mp3          # offline, as fast as possible
uv run grapevine replay ../data/samples/kickback-3p.mp3 --speed 1 # real time, to test latency
uv run grapevine replay ../data/samples/kickback-3p.mp3 --set turn_rules.max_turn_ms=10000
uv run grapevine report 3                                         # print and score run 3
uv run grapevine accept 3
uv run grapevine merge
```

A run with a new setting is a new run on the same session. If a clip has a `.labels.json` file, the report scores speaker accuracy, word error rate, and expected facts.

## Sample data

- `data/samples/kickback-3p` is a 2-minute clip with 3 speakers.
- `data/samples/corpus/` has 4 sessions of one friend group. They are 4 to 9 minutes long, with 3 to 8 speakers and 29 people in total.

Each sample has a `.script.json` file with the lines and the expected facts. The generator reads the script with macOS voices and writes the hand labels:

```sh
cd backend && uv run python -m grapevine.sample ../data/samples/corpus/pregame-4p.script.json
```

`data/grapevine.db` holds runs of the samples, so the views have data on the first start.

## Tests

```sh
cd backend
uv run pytest                          # unit tests with fake models
GRAPEVINE_E2E=1 uv run pytest -k e2e   # the full stack in Chrome, needs LM Studio
cd .. && npm run build                 # type check and build the client
```
