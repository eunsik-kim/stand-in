# Stand-In

Rehearse any situation in front of AI stand-ins for real people. Each character has a private goal, a secret, and a mood that changes. A director agent runs the scene. The camera and voice signals reach the characters, and the debrief reveals what they were really thinking.

## Run

```bash
uv run server.py        # http://localhost:8010, open in Chrome (speech recognition needs Chrome)
uv run play.py "상황 한 문장"                       # terminal version
uv run play.py "..." --auto "플레이어 성격"         # simulated player, for testing
uv run make_preset.py <id> "상황 한 문장"           # ready-made scene with background + portraits
```

`.env` needs `MODEL_API_KEY` (Meta Model API). Without it the client falls back to `CSR_VLM_API_KEY` (OpenRouter). Models can be overridden with `STANDIN_WRITER_MODEL`, `STANDIN_ACTOR_MODEL`, `STANDIN_CHECK_MODEL`, and `STANDIN_IMAGE_MODEL`.

## Flow

home (preset, or one sentence) → optional questions with recommended answers already filled in → scenario compiled and checked →
briefing: every field editable, background/portraits can be (re)generated, the user picks how much of the cast to see
(names only / profile / goals and secrets) → stage → debrief (everything revealed).

## Pieces

| file | role |
|---|---|
| `standin/intake.py` | situation → 4-6 tailored questions, each with a recommended answer, plus difficulty |
| `standin/voice.py` | natural speech per character through OpenRouter (Meta has no text-to-speech); needs `CSR_VLM_API_KEY` |
| `standin/art.py` | background and 2x2 expression sheets, shared by presets and the briefing screen |
| `standin/compiler.py` | situation → scenario spec (cast, secrets, events, success criteria) + GATE 1 review |
| `standin/engine.py` | one streamed call writes who answers and what they say; mood owned by code; GATE 2 leak guard; walk-outs |
| `standin/signals.py` | camera/voice numbers → what people in the room would notice (code-owned thresholds) |
| `standin/debrief.py` | criteria scoring, turning points, with turn citations checked |
| `static/index.html` | stage: portraits with expressions, per-character TTS, speech recognition, MediaPipe gaze in the browser |
| `presets/` | pre-built scenes; images in `presets/img/` |
