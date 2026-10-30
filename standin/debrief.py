"""After the scene: score against the success criteria and pick the moments
where the user's line changed what someone privately thought.

Every cited turn must exist in the transcript; a debrief that cites turns that
never happened is re-asked once, then the bad items are dropped.
"""

from pydantic import BaseModel

from .engine import Scene
from .llm import LLM


class CriterionScore(BaseModel):
    criterion: str
    met: bool
    turn: int
    why: str


class TurningPoint(BaseModel):
    turn: int
    character: str
    what_user_said: str
    what_they_thought: str
    try_instead: str


class Debrief(BaseModel):
    verdict: str
    criteria: list[CriterionScore]
    turning_points: list[TurningPoint]


def full_record(scene: Scene) -> str:
    s = scene.scenario
    secrets = "\n".join(f"- {c.name}: goal: {c.goal} | secret: {c.secret} | reveal_when: {c.reveal_when}"
                        for c in s.cast)
    lines = []
    for ln in scene.lines:
        lines.append(f"[{ln.turn}] {ln.speaker}: {ln.say}")
        if ln.inner:
            lines.append(f"    thought: {ln.inner} (mood {ln.mood})")
    return f"Characters' private side:\n{secrets}\n\nTranscript:\n" + "\n".join(lines)


def debrief(coach: LLM, scene: Scene, ui: str = "") -> Debrief:
    s = scene.scenario
    turns = {ln.turn for ln in scene.lines}
    ask = (
        f"Situation: {s.setting}\nUser played: {s.user_role}; goal: {s.user_goal}\n"
        f"Success criteria:\n" + "\n".join(f"- {c}" for c in s.success_criteria) + "\n\n"
        f"{full_record(scene)}\n\nScene ended: {scene.ended}\n\n"
        f"Write the debrief in language '{ui or s.language}', but keep quotes and try_instead in '{s.language}'. "
        "Be specific and honest, quote the transcript.\n"
        "- criteria: one entry per success criterion, with the turn number that decides it.\n"
        "- turning_points: 2-4 moments where a user line changed a character's private thinking or mood "
        "(for better or worse). Show what they thought, and a concrete line the user could have said instead.\n"
        'Return {"verdict": "2-3 sentences", "criteria": [{"criterion": "...", "met": true, "turn": 3, '
        '"why": "..."}], "turning_points": [{"turn": 3, "character": "...", "what_user_said": "...", '
        '"what_they_thought": "...", "try_instead": "..."}]}'
    )
    system = "You are a sharp, kind rehearsal coach. Reply with a single JSON object."
    d = coach.ask(Debrief, system, ask)
    bad = [x.turn for x in [*d.criteria, *d.turning_points] if x.turn not in turns]
    if bad:
        d = coach.ask(Debrief, system, ask + f"\n\nOnly cite turns that exist: {sorted(turns)}.")
        d.criteria = [x for x in d.criteria if x.turn in turns]
        d.turning_points = [x for x in d.turning_points if x.turn in turns]
    return d
