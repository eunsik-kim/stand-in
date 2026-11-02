# Demo video storyboard

Target 2:40, hard limit 3:00 (submission form: "3-Minute Demo Video").
One message: **rehearse the conversation before it happens**, and when it goes wrong, go back and say it differently.
Judged on creativity, technical ambition, and whether it works (indie track), so everything shown is the live site.

Story: "Pot Roast and Pink Slips" (en_dinner). You tell your parents you quit your job to start a company.
Mom (Diane) hides $8,000 she would lend you; Dad (Frank) once quit a county job at 32 and had to beg for it back;
brother Josh walks in at turn 2 and was passed over for a promotion last week.

| # | Time | Picture | Narration (AI voice) | On screen |
|---|---|---|---|---|
| 1 | 0:00-0:07 | Cold open: Frank, quietly, admits he quit at 32 and begged for his job back (from take 2) | none, his own voice | none |
| 2 | 0:07-0:11 | Title card | none | **Stand-In** / Rehearse the conversation before it happens |
| 3 | 0:11-0:26 | Three story photos, slow push-in: the dinner table, a resignation room, an investor pitch | "Some conversations you only get one shot at. Telling your parents you quit your job. Resigning. Pitching investors. You can't practice on the real people, and a chatbot just agrees with you." | one label per photo, under 6 words |
| 4 | 0:26-0:42 | Home grid, pick the dinner story, the briefing with the cast (zoom on the cast) | "Stand-In fills the room with AI stand-ins. Each one has a goal, a secret, and a mood that moves with every word you say." | |
| 5 | 0:42-1:15 | Take 1: a blunt first line, Mom and Dad push back in their own voices, Josh walks in with a joke. Middle turns at 4x. Ending page: what you never found out | "My first try." ... "When it ends, you see what they were hiding, and the moment it turned." | "4x" badge on the fast part |
| 6 | 1:15-1:50 | "Go back to that moment", a better line to Dad, his secret comes out (the cold open, now in context), the clue lands, a goal checks off, ending: won | "So go back to that moment, and say it differently." | zoom on the clue and the goal toast |
| 7 | 1:50-2:12 | Make your own: type one line, about a minute later (sped up) a new scene with photos; quick cuts of a mystery, a piece of history, a first date; the language picker | "Rehearse anything from one sentence, or step into a mystery, a piece of history, a first date, in English, Korean or Japanese." | "about 1 minute, sped up" |
| 8 | 2:12-2:35 | How it is built: a simple diagram card, parts lighting up as they are named | "Under the hood, one streamed Muse Spark call plays the director and every character, so the first words land in about three seconds. Code owns the moods, and nudges people to open up as trust grows. Muse Image paints the scene, then edits that same photo as the story turns, so everyone keeps the same face." | Muse Spark / Muse Image labels |
| 9 | 2:35-2:42 | End card | "Stand-In. Try it at stand-in dot fly dot dev." | stand-in.fly.dev, GitHub link |

## Production

- **Capture**: headless browser on the live site, frames from the DevTools screencast, every character voice caught as it plays. Input is typed (no mic or camera in a headless browser).
- **Takes**: the server caches every model reply by request, so a dry run finds lines that work and the final take replays them without waiting. Take 2 starts from "Go back to that moment" on take 1's ending.
- **Edit** (`demo/edit.py`): real-time where it matters, 4x on filler turns, waits for the model cut; smooth 1.5-2x zoom on what is being clicked (positions logged by the recorder); cross-fades between scenes; narration ducks the character voices.
- **Narration**: ElevenLabs through the same OpenRouter key the app uses, one calm voice, generated per line so timing can be tuned.
- **Cards** (title, problem photos, diagram, end): HTML rendered at 1920x1080, so text stays sharp.
- **Checks before export**: total under 3:00, no clipped audio, every claim matches the code (first sentence about 3 s in the server logs, mood clamped in code, each turning-point photo edited from the last one in `standin/art.py`).
