"""The last page of a scene: how it ended, told back to the player like a game's ending."""

from pydantic import BaseModel

from .debrief import full_record
from .engine import Scene
from .llm import LLM


class Ending(BaseModel):
    title: str
    story: str
    met: list[bool]
    best_line: str = ""


def ending(writer: LLM, scene: Scene, ui: str) -> Ending:
    s = scene.scenario
    return writer.ask(
        Ending, "You write the closing page of an interactive story. Reply with a single JSON object.",
        f"Setting: {s.setting}\nThe player was: {s.user_role}\nTheir goal: {s.user_goal}\n"
        "Outcomes to judge, in order:\n" + "\n".join(f"- {c}" for c in s.success_criteria)
        + f"\n\n{full_record(scene)}\n\nThe scene ended: {scene.ended}\n\n"
        f"Write in language '{ui or s.language}'.\n"
        "title: the name of the ending this player reached, 2-5 words, specific to what happened, like a game ending.\n"
        "story: 2-3 short sentences saying what became of everyone after the scene, addressed to the player as 'you'. "
        "Follow from what was actually said; if things were left open, close them the way this conversation pointed. No advice.\n"
        "met: true or false for each outcome, judged only from the transcript.\n"
        "best_line: the player's own line that changed things most, copied exactly, or empty if none did.\n"
        'Return {"title": "...", "story": "...", "met": [true, false], "best_line": "..."}',
    )
