"""Narration for the demo video, one clip per line, through the app's own voice path (ElevenLabs on OpenRouter).

Clips are cached by text, so changing one line only remakes that line.

Usage: uv run python -m demo.narrate   -> demo/out/vo/<id>.mp3 and demo/out/vo/lines.json (durations)
"""
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from standin import voice
from standin.llm import Spend

NARRATOR = "alice"  # the dinner cast speaks with sarah, george and brian
LINES = {
    "problem": "Some conversations you only get one shot at. Telling your parents you quit your job. Resigning. "
               "Pitching investors. You can't practice on the real people, and a chatbot just agrees with you.",
    "solution": "Stand-In fills the room with AI stand-ins. Each one has a goal, a secret, "
                "and a mood that moves with every word you say.",
    "first": "First try. I just say it.",
    "reveal": "When it ends, you see what they were hiding, and the moment it turned.",
    "back": "So go back to that moment, and say it differently.",
    "breadth": "Rehearse anything from one sentence. Or step into a mystery, a piece of history, a first date. "
               "In English, Korean or Japanese.",
    "built": "Under the hood, one streamed Muse Spark call plays the director and every character, "
             "so the first words land in about three seconds. Code owns the moods, and nudges people to open up as trust grows. "
             "Muse Image paints the scene, then edits that same photo as the story turns, so everyone keeps the same face.",
    "cta": "Stand-In. Try it at stand-in dot fly dot dev.",
}


def main():
    load_dotenv()
    out = Path("demo/out/vo")
    out.mkdir(parents=True, exist_ok=True)
    spend, durations = Spend(2.0), {}
    for key, text in LINES.items():
        path = out / f"{key}.mp3"
        path.write_bytes(voice.synthesize(text, NARRATOR, spend, out))
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                               capture_output=True, text=True, check=True)
        durations[key] = round(float(probe.stdout.strip()), 2)
        print(f"{key:9} {durations[key]:5.1f}s  {text[:60]}")
    (out / "lines.json").write_text(json.dumps(durations, indent=1))
    print(f"total {sum(durations.values()):.1f}s, ${spend.usd:.3f}")


if __name__ == "__main__":
    main()
