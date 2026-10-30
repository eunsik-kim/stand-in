"""Before compiling: a few questions that make the scene match the user's real situation.

Every question comes with a recommended answer, so the user can accept all the
defaults and still get a sensible scene, or replace any of them with their own
facts. Difficulty is a fixed question owned by code.
"""

from pydantic import BaseModel, Field

from .llm import LLM


class Question(BaseModel):
    id: str
    ask: str
    hint: str = Field(default="", description="why this matters, one short clause")
    suggested: str = Field(description="a plausible recommended answer the user can keep")
    options: list[str] = Field(default=[], description="2-4 quick picks when a short choice works")


class Intake(BaseModel):
    questions: list[Question]


LEVELS = {
    "en": ("How tough should it be?", "how easily they open up", ["Easy", "Realistic", "Tough"]),
    "ko": ("난이도는 어느 정도로 할까요?", "상대가 얼마나 마음을 쉽게 여는지", ["편하게", "현실적으로", "빡세게"]),
    "ja": ("難易度はどうしますか?", "相手がどれだけ心を開きやすいか", ["やさしめ", "現実的", "厳しめ"]),
}
DIFFICULTY = {label: level for _, _, labels in LEVELS.values() for label, level in zip(labels, ("easy", "real", "hard"))}


def difficulty_question(ui: str) -> Question:
    ask, hint, labels = LEVELS.get(ui, LEVELS["en"])
    return Question(id="difficulty", ask=ask, hint=hint, suggested=labels[1], options=labels)


def questions(llm: LLM, situation: str, ui: str = "en") -> list[Question]:
    got = llm.ask(
        Intake, "You prepare rehearsal scenes. Reply with a single JSON object and nothing else.",
        f'The user wants to rehearse this situation:\n"{situation}"\n\n'
        "Before the scene is built, ask 4-6 short questions whose answers make the rehearsal match their real case. "
        "Cover, unless the sentence already says it:\n"
        "- facts about the user they will lean on (their numbers, achievements, constraints, history with these people)\n"
        "- who exactly will be there and what they are like (personality, relationship, what they care about)\n"
        "- what the user already knows about the other side's position\n"
        "- the outcome they want and the least they would accept\n"
        "- the moment they are most worried about\n"
        f"Write in language '{ui}'. For each question give `suggested`: a concrete, plausible answer "
        "for a typical person in this situation (the user may keep it), and `options` (2-4 short picks) when a quick choice fits.\n"
        'Return {"questions": [{"id": "short_snake_id", "ask": "...", "hint": "...", "suggested": "...", "options": []}]}',
    )
    return got.questions[:6] + [difficulty_question(ui)]
