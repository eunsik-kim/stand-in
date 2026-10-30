"""Live coach: after each user line, what is done, what to try next, how to say it better.

It sees only what the user could see (goal, transcript), never the characters'
goals or secrets, so its help cannot spoil the scene. It runs beside the scene
call, so the user never waits for it.
"""

from typing import Literal

from pydantic import BaseModel, Field

from .engine import Scene
from .llm import LLM


class GoalState(BaseModel):
    status: Literal["todo", "partial", "done"]
    note: str = Field(default="", description="what happened, a few words")


class Coaching(BaseModel):
    goals: list[GoalState]
    good: str = Field(default="", description="what worked in their last line, only if something really did")
    hint: str = Field(description="the single next move")
    better: str = Field(default="", description="their last line said better, in the scene's language")
    why: str = Field(default="", description="why that is better")


def coach_turn(llm: LLM, scene: Scene, said: str, goals: list[str], ui: str) -> Coaching:
    s = scene.scenario
    criteria = "\n".join(f"{i + 1}. [{goals[i]}] {c}" for i, c in enumerate(s.success_criteria))
    foreign = s.language != ui
    return llm.ask(
        Coaching, "You are a sharp, warm coach whispering to someone during a live conversation. Reply with a single JSON object.",
        f"They play: {s.user_role}\nTheir goal: {s.user_goal}\nWhat they know: {'; '.join(s.user_knows)}\n\n"
        f"Transcript:\n{scene.public()}\n\nTheir latest line: \"{said}\"\n\n"
        f"Goals with current status:\n{criteria}\n\n"
        "goals: one entry per goal, in order. done = clearly achieved in the transcript; partial = real progress; "
        "todo = not yet. A goal already marked done stays done. Be honest, not generous.\n"
        "good: name the specific thing that worked in the latest line, or leave it empty.\n"
        "hint: the one next move, at most 15 words, concrete to this conversation. Do not guess what others are hiding.\n"
        "better: rewrite the latest line the way a skilled person would say it, keeping their facts and intent, "
        f"in language '{s.language}'. Leave it empty if the line was already good.\n"
        + ("If their line had language mistakes, fix them in `better` and explain the main one in `why`.\n" if foreign else "")
        + f"Write note, good, hint and why in language '{ui}'.\n"
        'Return {"goals": [{"status": "todo|partial|done", "note": "..."}], "good": "", "hint": "...", "better": "", "why": ""}',
    )
