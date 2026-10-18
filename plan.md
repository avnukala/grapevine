# Grapevine plan

Grapevine listens to a group conversation and builds a knowledge graph of the people in it. The graph shows who knows whom, how they met, the stories they tell, and the groups they belong to. A timeline shows when each relationship and story happened. The MVP is one live kickback with up to 8 talkers on one laptop, with all data kept for later sessions.

## Priorities

1. Capture the right data: speaker detection, transcription, and fact extraction. Speaker detection must be accurate. A wrong speaker label puts facts on the wrong people.
2. Make every result reproducible and easy to inspect with replays of recorded audio.
3. Make the views good enough to check the data. Polish the visuals after the data is right.

The first milestone is a recorded MP3 that replays through the pipeline and gives a good graph. Start with 2 or 3 people talking. The MVP goal is a live session with 8 talkers where Grapevine captures most data, even if parts look rough.

## What exists today

- A React and Vite front end, with an Express server that calls any OpenAI-compatible model.
- Text input only. The user types text and presses Cmd+Enter. The server sends the current graph and the new text to the model.
- The model returns graph operations: add, update, or remove a person, and add, update, remove, or retype a relationship. The browser applies them.
- People have a name, aliases, gender, role, org, and notes. Gender sets the node color.
- Relationships have a category, a short label, a description, a direction, and a strength from 1 to 5.
- Cytoscape renders the graph. Changed elements flash. There is a change log, undo, reset, and a person details panel.
- The graph lives in browser `sessionStorage`. It does not last after the tab closes.

## Session flow

1. Introductions. Each person in the room says their name and a few sentences. Grapevine saves a voice embedding for each person and links them to a person in the `people` table. If the name matches more than one person, the user picks from a list or adds a nickname.
2. First prompts. Grapevine suggests openers between people in the room, for example "Ray and Jasper, how did you meet?".
3. Free conversation. People talk. Grapevine extracts facts, updates the graph, and shows what it captured.
4. Corrections. People say "no, that is wrong" or add detail. The extractor adds a new fact that replaces the old one.
5. End of session. A larger model writes stories and relationship summaries from the facts of the session.

## Architecture

- A Python back end owns all server work: audio input, transcription, speaker detection, model calls, the graph, and SQLite.
- The back end pushes each change to the client over a WebSocket.
- The React client only renders views and sends user actions, such as corrections and name picks. Node only builds and serves the client.
- Everything runs on the laptop. Model calls go to a local OpenAI-compatible server, such as LM Studio. No data goes to a hosted API.
- Each model sits behind a small interface, so a test can swap it: transcriber, speaker detector, voice embedder, extraction model, and summary model.
- Audio input reads from the microphone or from a file. Live sessions and replays share one code path.
- The server keeps the graph in memory and writes each change to SQLite at once.

## Audio pipeline

- Speaker detection uses NVIDIA Nemotron 3 Diarization. It runs live and offline, and it scored best on the Voice Arena diarization benchmark. It handles up to 8 speakers, so the MVP allows 8 talkers in the room.
- Nemotron numbers speakers in the order they first talk. During introductions, speaker 0 is the first person who introduces themselves.
- Each speech segment also gets a voice embedding. The pipeline compares it to the embeddings saved during introductions to confirm the name.
- If no saved voice is close enough, the pipeline marks the segment unknown. The extractor still uses it, and the UI flags it so someone can say who spoke.
- Transcription uses a local Whisper model. Rough transcripts are fine, because the extraction model can fill gaps.
- Test Nemotron on the M5 Mac first. If it does not run well, fall back to DiariZen for offline runs and pyannote Community-1 for live runs.
- The pipeline sends one speaker turn at a time to the extractor. It splits turns longer than about 15 seconds at a sentence end. The splitting rules are a setting that tests can change.

## Extraction

- A small, fast local model extracts facts live. The target is 2 to 3 seconds from the end of speech to the fact on screen.
- The transcript line and the speaker glow show at once, before extraction finishes.
- Each speaker turn goes to the model with the speaker name. The model maps "I" and "me" to that speaker.
- The first version sends the whole graph with each request. A later MVP task sends only the relevant part, and a test checks that the output stays the same.
- The model groups facts from one long story so the story becomes one record. The model decides how to group them. The end-of-session pass writes the final story.
- The model matches names to people by name and aliases. If a name could mean more than one person, the prompt panel asks "Which Andrew, Gao or Moy?" and the fact waits for the answer.
- Facts are append-only. A correction adds a new fact, and the old fact points to it with `revised_by_fact_id`.
- If two people disagree, the model keeps both facts, marks them disputed, and the prompt panel asks the room.
- A second worker, the prompter, runs in parallel. It reads the graph and recent turns and writes questions for the prompt panel.
- The end-of-session pass uses a larger local model. It writes stories and relationship summaries from the facts of the run.

## Data model

A **session** is one audio recording. A **run** is one pass of the pipeline over a session with a fixed configuration. A live event is one run. Each replay with a new model, prompt, or setting is a new run on the same audio, so runs never overwrite each other.

People are global. Every run reads the same `people` table, and facts from every run point to it. To view one session, filter facts by run. A new session still starts with an empty graph on screen, because the screen shows only the facts of the current run.

```
sessions        id, name, recorded_at, audio_uri (shared Drive path), notes
runs            id, session_id, kind (live | replay | merge), config_json, code_version,
                accepted, created_at
model_calls     id, run_id, worker (extractor | prompter | summarizer), model, model_file_hash,
                prompt_version, seed, input_hash, request_json, response_json, latency_ms
changes         id, run_id, seq, audio_ms, segment_id, model_call_id, op_json

segments        id, run_id, start_ms, end_ms, speaker_label, person_id, speaker_confidence, text
speakers        run_id, speaker_label, person_id, voice_embedding

people          id, name, aliases, gender, created_at

facts           id, run_id, kind, text, detail, told_by_person_id, when_text, start_date,
                end_date, date_precision, event_id, confidence, disputed, revised_by_fact_id,
                created_at
fact_people     fact_id, person_id, role
fact_segments   fact_id, segment_id

events          id, run_id, name, when_text, start_date, end_date, date_precision, place
event_people    event_id, person_id, role

groups          id, name, aliases, description
group_members   group_id, person_id, since, until, source_fact_id

relationships   id, run_id, person_a_id, person_b_id, category, label, summary,
                status (current | past), met_on, met_event_id, closeness, recency, sentiment,
                updated_at
relationship_evidence  relationship_id, fact_id

prompts         id, run_id, text, reason, person_ids, shown_at_ms, dismissed_at_ms
```

Rules for the data:

- Fact kinds are `personal`, `relationship`, `story`, `membership`, and `correction`.
- `fact_people` links a fact to each person in it. The `role` field says how, for example subject or mentioned.
- `when_text` stores the date as people said it, for example "freshman year". The model fills `start_date`, `end_date`, and `date_precision` (year, season, month, or day) when it can. The timeline only shows resolved dates.
- An event, such as "Electric Zoo 2022", is one row with one date range. Many facts link to one event. Events do not nest in the MVP.
- Two people can have more than one relationship row, one per category. Each row has a status. For example, Jasper and Sophie have a `romantic` row with status `past` and a `friend` row with status `current`.
- Relationship categories are `romantic`, `family`, `friend`, `professional`, and `other`. Remove `acquaintance`.
- A relationship stores raw inputs for edge weight: the date they met, closeness, recency, and sentiment. The client computes the weight.
- Each relationship links to the facts that support it. The model decides when a new fact changes a relationship and when to recompute it from all its facts.
- Gender stays on `people` and still sets the node color.

### Global graph

- The global graph uses facts from runs marked `accepted`, at most one run per session.
- A `merge` run reads those facts and writes the global relationship summaries.
- The merge shows only conflicts for manual review: facts that disagree and people the model cannot match with high confidence. In review, the user keeps one fact, keeps both as disputed, or removes an edge.
- Live sessions do not read facts from past sessions. They read only the `people` table.

## Client views

### Graph view

- Move rendering from Cytoscape to D3.js.
- People in the room start at fixed points and get high inertia. They drift toward their real position over time and do not jump.
- When someone names a person who is not in the room, that person appears on the outside with low inertia.
- When someone talks about a relationship, its edge lights up. The newest fact shows under that edge for a few seconds, for example "met in 2022".
- The node label is the most common alias that is unique, for example "Gao" when two Andrews exist.
- The node of the person who talks glows.

### Timeline view

- A second view that shows events and dated facts on a time axis.
- When someone talks about an event, the timeline moves to its date and shows the new facts.
- The user switches between graph view and timeline view with a button. The app never switches by itself.
- The main use in the MVP is to check that the captured dates and events are correct. Design the layout when work on the view starts.

### Prompt panel

- A corner panel with no more than 2 questions at a time.
- Prompts are suggestions. The conversation does not need to follow them.
- Show a prompt after a pause of about 7 seconds or when facts conflict.
- Also show a prompt when a fact needs a date, or when a name could mean more than one person.
- Prefer people and pairs that have not had much time, so a few people do not take over the session.

### Debug and replay screen

- One screen that runs only in dev or test mode. The live view never shows it.
- It has an audio player with speed control and a scrubber, the transcript segments with speakers, the model calls, and the change log.
- It shows the graph and timeline at the current audio time.
- It shows the time for each step: speaker detection, transcription, and extraction.
- The current change log moves here.

## Edge weight

Edge weight drives the graph physics, so close relationships sit close together. The hope is that groups then form clusters without extra work.

The weight is a weighted sum of the raw inputs on each relationship. Tune the weights in the client and compare the graphs. Factors to try:

- Years known, from the date they met. Start with the largest weight, for example 60%.
- Recency: how close they are now.
- Closeness and the tone of their stories.
- Relationship category.

Clusters alone will not find named groups. Stories favor the people in the room, and some people come up only once. Named groups from the conversation stay the main source of group membership.

## Testing and reproducibility

- Record sample conversations with a few friends and one microphone. Store audio in the shared Drive folder. Commit the SQLite file to the private repo.
- Replay in two modes. Real-time mode streams the file at 1x or faster to test latency. Offline mode processes the file as fast as it can and saves each change with its audio time for the replay screen.
- Build both modes together.
- Label one clip of about 3 minutes by hand: who spoke when, and the expected facts. Use it to score speaker detection models and extraction prompts.
- For other clips, review the output by hand. A local model can act as judge to compare two runs.
- Use temperature 0 and a fixed seed. Record the model file hash and prompt version on each model call.
- Store every model call. A replay reuses a stored response when the input hash matches, so an unchanged replay gives the same result.
- Compare speaker detectors, transcribers, extraction models, prompts, and splitting rules as runs on the same session.

## Open questions

- How to show graph, timeline, and groups together.
- How to show groups: as clusters in the graph view, as a separate view where each node is a group, or both.
- The weights for edge weight. Decide by experiment.
- Which small local model gives the best facts within 2 to 3 seconds.
- How the model groups facts from one long story.
- Which local tool stores model traces, for example a local Langfuse. Connect it to coding agents later through MCP.

## Future improvements

- A larger background model that reads a large part of the graph and picks the context for the small live model. It can take 10 to 20 seconds per pass.
- Eras and nested events. Events such as "Cornell college era" hold other events, and the timeline shows them as brackets.
- Late arrivals. Someone who joins after introductions shows as "unknown speaker N". The user names them, Grapevine saves their voice, and the extractor re-runs their earlier segments.
- More than 8 talkers, with a speaker detector that supports more speakers.
- Voice commands to change the view, for example "go to the timeline view".
- Show facts from past sessions during a live session.
- A mobile web page to record a conversation anywhere, for example at a pregame, and process it later.
- Show the graph on a TV during the event.
- Negative edges that push two people apart.
- Concurrent "breakout room" conversations. Each small group has one microphone and builds the same graph at the same time, so the app must handle conflicts and concurrent writes.
