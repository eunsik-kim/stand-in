"""The last page of a scene: how it ended, told back to the player like a game's ending."""

from pydantic import BaseModel

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
        + f"\n\nEverything that was said and happened:\n{scene.public(viewer='record')}\n\n"
        f"Write in language '{ui or s.language}'.\n"
        "title: the name of this ending, 2-5 words, like an ending in a game, not a headline.\n"
        "story: 2-3 short sentences saying what became of everyone after the scene, addressed to the player as 'you'. "
        "Use only what is above: invent no facts, names or objects, and no advice.\n"
        "met: true or false for each outcome. It is true only if it happened in the transcript through the player's own side; "
        "if someone else did it, or you are unsure, it is false.\n"
        "best_line: the player's own line that changed things most, copied exactly, never one marked (auto); empty if none did.\n"
        'Return {"title": "...", "story": "...", "met": [true, false], "best_line": "..."}',
    )
