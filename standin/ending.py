"""The last page of a scene: how it ended, told back to the player like a game's ending."""

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from .debrief import full_record
from .engine import Scene
from .llm import LLM


class Ending(BaseModel):
    title: str
    story: str
    best_line: str = ""


class Judged(BaseModel):
    met: list[bool]
    came_out: list[str] = []


def judged(writer: LLM, scene: Scene) -> Judged:
    s = scene.scenario
    return writer.ask(
        Judged, "You judge a finished story strictly. Reply with a single JSON object.",
        "Outcomes to judge, in order:\n" + "\n".join(f"- {c}" for c in s.success_criteria)
        + f"\n\n{full_record(scene)}\n\n"
        "met: true or false for each outcome. It is true only if it happened in the transcript through the player's own side; "
        "if someone else did it, or you are unsure, it is false.\n"
        "came_out: the names of people whose secret was actually stated out loud in the transcript, by anyone. A lie or a hint does not count.\n"
        'Return {"met": [true, false], "came_out": ["..."]}',
    )


def written(writer: LLM, scene: Scene, ui: str) -> Ending:
    s = scene.scenario
    return writer.ask(
        Ending, "You write the closing page of an interactive story. Reply with a single JSON object.",
        f"Setting: {s.setting}\nThe player was: {s.user_role}\nTheir goal: {s.user_goal}\n"
        f"\nEverything that was said and happened:\n{scene.public(viewer='record')}\n\n"
        f"Write in language '{ui or s.language}', addressing the player as 'you' (in Korean 당신, with -습니다 endings).\n"
        "title: the name of this ending, 2-5 words, like an ending in a game, not a headline.\n"
        "story: 2-3 short sentences. First what actually came out in this scene and what the player did, naming the central event; "
        "then what that leads to. Do not list where people are or what they hold. Invent no facts, names or objects, and give no advice.\n"
        "best_line: the player's own line that changed things most, copied exactly, never one marked (auto); empty if none did.\n"
        'Return {"title": "...", "story": "...", "best_line": "..."}',
    )


def ending(writer: LLM, scene: Scene, ui: str) -> tuple[Ending, Judged]:
    with ThreadPoolExecutor(2) as pool:
        words, verdict = pool.submit(written, writer, scene, ui), pool.submit(judged, writer, scene)
    return words.result(), verdict.result()
