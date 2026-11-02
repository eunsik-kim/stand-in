"""The last page of a scene: how it ended, told back to the player like a game's ending."""

from pydantic import BaseModel

from .debrief import full_record
from .engine import Scene
from .llm import LLM


class Ending(BaseModel):
    title: str
    story: str
    best_line: str = ""
    pivot_turn: int = 0


class Truth(BaseModel):
    name: str
    line: str


class Judged(BaseModel):
    came_out: list[str] = []
    met: list[bool]
    truths: list[Truth] = []


def judged(writer: LLM, scene: Scene, ui: str) -> Judged:
    s = scene.scenario
    return writer.ask(
        Judged, "You judge a finished story strictly. Reply with a single JSON object.",
        "Outcomes to judge, in order:\n" + "\n".join(f"- {c}" for c in s.success_criteria)
        + f"\n\n{full_record(scene)}\n\n"
        "came_out: the names of people whose secret was actually stated out loud in the transcript, by anyone. A lie or a hint does not count.\n"
        "met: true or false for each outcome. It is true only if it happened in the transcript through the player's own side. "
        "If someone else did it, it is false.\n"
        f"truths: for every person, one plain sentence in language '{ui or s.language}' saying what was really going on with them, naming them, "
        "stating the fact only, written for the player to read afterwards. No notes about how to act, and nothing about what they wanted.\n"
        'Return {"came_out": ["..."], "met": [true, false], "truths": [{"name": "...", "line": "..."}]}',
    )


def written(writer: LLM, scene: Scene, ui: str, met: list[bool]) -> Ending:
    s = scene.scenario
    return writer.ask(
        Ending, "You write the closing page of an interactive story. Reply with a single JSON object.",
        f"Setting: {s.setting}\nThe player was: {s.user_role}\nTheir goal: {s.user_goal}\n"
        "Outcomes, already decided, and the story must agree with them:\n"
        + "\n".join(f"- {c}: {'reached' if m else 'not reached'}" for c, m in zip(s.success_criteria, met))
        + f"\n\nEverything that was said and happened:\n{scene.public(viewer='record')}\n\n"
        f"Write in language '{ui or s.language}', addressing the player as 'you' (in Korean 당신, with plain -다 endings and no honorific 시 on others).\n"
        "title: the name of this ending, 2-5 words, like an ending in a game, not a headline.\n"
        "story: 2-3 short sentences. First what actually came out and what the player really did, jokes and failures included; "
        "then what that leads to. Name the central event. Never praise the player, end on no moral, and do not list where people are. "
        "Invent no facts, names or objects, give no advice, and do not quote lines inside the sentences. Write 'you', never 'user'.\n"
        "best_line: the player's own line that changed things most, copied exactly, never one marked (auto); empty if none did.\n"
        "pivot_turn: the number in brackets of the player's own turn, before the last two, where a different move would have changed the most. Never a line in parentheses.\n"
        'Return {"title": "...", "story": "...", "best_line": "...", "pivot_turn": 3}',
    )


def ending(writer: LLM, scene: Scene, ui: str) -> tuple[Ending, Judged]:
    done = judged(writer, scene, ui)
    if scene.verdict is not None:  # the last turn was played from this verdict, so the page keeps it
        done.met = scene.verdict
    return written(writer, scene, ui, done.met), done
