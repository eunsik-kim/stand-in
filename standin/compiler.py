"""One sentence (+ optional material) -> a playable Scenario.

GATE 1: the scenario must be practice, not a chat. Code checks the structure;
a separate checker model checks that every character pushes against the
user's goal and that each success criterion can be judged from the dialogue.
Failures go back to the writer as repair notes.
"""

import re

from pydantic import BaseModel

from .llm import LLM
from .spec import Scenario

WRITER_SYSTEM = (
    "You design rehearsal scenarios: realistic situations where a person practices facing other people. "
    "You write characters who behave like real humans with their own interests, not helpful assistants. "
    "Reply with a single JSON object and nothing else."
)

SHAPE = (
    '{"title": "...", "language": "ko|en|ja|...", "setting": "...", "user_role": "...", "user_goal": "...", "user_knows": ["..."], '
    '"cast": [{"name": "...", "role": "...", "appearance": "...", "voice": "...", "gender": "m|f", "goal": "...", '
    '"secret": "...", "reveal_when": "...", "red_lines": ["..."], "mood": {"trust": 0-100, "patience": 0-100}}], '
    '"opening_speaker": "<cast name>", "opening_line": "...", "phases": ["..."], '
    '"events": [{"at_turn": 3, "description": "..."}], "success_criteria": ["..."], "max_turns": 12}'
)


DIFFICULTY = {
    "easy": "easy: people are fairly open (trust 60-75, patience 70+), reveal_when is a reasonable effort, one mild event.",
    "real": "realistic: friction as it would really be for this situation, patience 45 or more.",
    "hard": "hard: guarded people (trust 25-45, patience 35-50), reveal_when needs real skill, events add strong pressure.",
}


class Review(BaseModel):
    problems: list[str]


def facts(text: str) -> set[str]:
    """Number+unit tokens ('4%', '80만원', '120곳') used to spot a secret repeated in public text."""
    return {re.sub(r"\s+", "", m) for m in re.findall(r"\d[\d,.~]*\s*(?:%|[가-힣A-Za-z]{1,2})", text)}


def structural_problems(s: Scenario, told: str = "") -> list[str]:
    names = [c.name for c in s.cast]
    p = []
    if not 1 <= len(s.cast) <= 4:
        p.append(f"cast must have 1-4 characters, has {len(s.cast)}")
    if len(set(names)) != len(names):
        p.append("character names must be unique")
    if s.opening_speaker not in names:
        p.append(f"opening_speaker '{s.opening_speaker}' is not in the cast {names}")
    if not 2 <= len(s.success_criteria) <= 4:
        p.append("give 2-4 success criteria")
    for e in s.events:
        if e.at_turn >= s.max_turns:
            p.append(f"event at turn {e.at_turn} never happens (max_turns {s.max_turns})")
    public = [("opening_line", s.opening_line), ("user_role", s.user_role), ("user_knows", " ".join(s.user_knows))]
    public += [(f"event at turn {e.at_turn}", e.description) for e in s.events]
    for c in s.cast:
        for where, text in public:
            shared = (facts(c.secret) & facts(text)) - facts(told)  # the user already knows what they told us
            if shared:
                p.append(f"{where} gives away {c.name}'s secret ({', '.join(sorted(shared))}); keep secrets out of public text")
    return p


def review(checker: LLM, s: Scenario, told: str) -> list[str]:
    return checker.ask(
        Review, "You are a strict reviewer of rehearsal scenarios. Reply with a single JSON object.",
        f"What the user told us:\n{told}\n\nScenario JSON:\n{s.model_dump_json(indent=1)}\n\n"
        "List concrete problems, only these kinds:\n"
        "1. A character whose private goal does not create any friction with the user's goal (they would just agree).\n"
        "2. A secret that is vague or would not change how the user should play (it must be a concrete fact).\n"
        "3. A reveal_when that is not something the user can actually do or say in this conversation.\n"
        "4. A success criterion that cannot be judged from the dialogue transcript alone.\n"
        "5. Generic content that would fit any scenario instead of this situation.\n"
        "6. A secret stated or strongly hinted in the opening_line, an event, user_role or user_knows.\n"
        "7. Facts about the user (achievements, numbers, history, offers) in user_role or user_knows that the user did not tell us.\n"
        "8. A reveal_when that requires several separate things at once instead of one clear move "
        "(count the actions it lists; more than one is a problem).\n"
        "Do not invent problems; a good scenario gets an empty list.\n"
        'Return {"problems": ["..."]} with an empty list if there are none.',
    ).problems


def told_text(situation: str, answers: list[tuple[str, str]], material: str = "") -> str:
    out = f'Situation: "{situation}"'
    for q, a in answers:
        if a.strip():
            out += f"\nQ: {q}\nA: {a.strip()}"
    if material:
        out += f"\n\nMaterial the user provided:\n{material[:12000]}"
    return out


def compile_scenario(writer: LLM, checker: LLM, situation: str, material: str = "",
                     answers: list[tuple[str, str]] = (), difficulty: str = "real", ui: str = "",
                     repairs: int = 2, log=print) -> tuple[Scenario, list[dict]]:
    """Returns the scenario and the gate history (problems found per attempt)."""
    told = told_text(situation, list(answers), material)
    ask = (
        f"The user wants to rehearse this. Everything they told us:\n{told}\n\n"
        "Design the scene. Rules:\n"
        "- language: what the people in the scene actually speak (a Tokyo cafe is 'ja' even if the request is in Korean); otherwise the user's language. opening_line and all spoken lines use only that language.\n"
        + (f"- Everything else (title, setting, roles, goals, user_knows, phases, events, success_criteria, character notes) is read by the user: write it in language '{ui}'.\n" if ui else "")
        + 
        "- The user plays themselves. user_role and user_knows use only facts the user told us; never invent their achievements, numbers, history or offers. "
        "If something is unknown, leave it general: the user will bring their own facts while speaking.\n"
        "- user_knows: 2-4 facts the user really walks in knowing (from their answers first). Never part of any secret.\n"
        "- Include only the people who would really be there (1-4). Give them real names fitting the setting. "
        "Use what the user said about them (personality, relationship).\n"
        "- Each character has a private goal that pushes against the user's goal in a different way, "
        "and exactly ONE concrete secret (a number, a fact, a constraint) that would change the user's strategy if known.\n"
        "- reveal_when: ONE specific move the user can make in the conversation that earns the secret, "
        "e.g. 'asks directly whether an exception process exists'. Not a chain like 'does A, then B, and asks C'.\n"
        "- Secrets never appear in the opening_line, events, user_role or user_knows.\n"
        f"- Difficulty {DIFFICULTY.get(difficulty, DIFFICULTY['real'])}\n"
        "- 1-3 events that add pressure (interruption, time, new public information).\n"
        "- success_criteria: 2-4 outcomes judgeable from the transcript, tied to what the user said they want. "
        "Each is one clear thing a well-prepared person can really achieve within this conversation, mostly about what the user does.\n"
        "- opening_line: the first thing said, in character, setting the tone.\n"
        f"Return:\n{SHAPE}"
    )
    history = []
    scenario = writer.ask(Scenario, WRITER_SYSTEM, ask)
    for attempt in range(repairs + 1):
        problems = structural_problems(scenario, told)
        if not problems:
            problems = review(checker, scenario, told)
        history.append({"attempt": attempt, "problems": problems})
        log(f"  gate 1 attempt {attempt}: {len(problems)} problems" + "".join(f"\n    - {p}" for p in problems))
        if not problems or attempt == repairs:
            break
        scenario = writer.ask(
            Scenario, WRITER_SYSTEM,
            f"{ask}\n\nYour previous version:\n{scenario.model_dump_json()}\n\n"
            "A reviewer found these problems. Fix all of them and return the full corrected JSON:\n"
            + "\n".join(f"- {p}" for p in problems),
        )
    return scenario, history
