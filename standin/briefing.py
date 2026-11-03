"""The few lines that walk the player into a scene: where they are, who they are, what's been going on,
who is in front of them and what they want out of it, said the way a friend would explain it.

Written from what the player may know only (never a goal or secret of the cast), in the player's language,
and cached like every other reply, so a ready-made story is briefed once.
"""
from pydantic import BaseModel

from standin.llm import LLM
from standin.spec import Scenario

VOICE = {  # the language, its tone, and how the last sentence states the goal
    "ko": "Korean only, in warm, plain 해요체 ('지금은 ~예요', '~하고 있어요'), the way a friend explains before a game starts; "
          "the last sentence states the goal as the player's own aim, e.g. '~하는 게 목표예요'",
    "en": "English only, warm and plain, the way a friend explains before a game starts; "
          "the last sentence states the goal as the player's own aim, e.g. 'Your goal is to ...'",
    "ja": "Japanese only, in gentle です/ます, the way a friend explains before a game starts; "
          "the last sentence states the goal as the player's own aim, e.g. '目標は〜することです'",
}


class Briefing(BaseModel):
    lines: list[str]


def briefing(writer: LLM, s: Scenario, ui: str) -> list[str]:
    here = [c for c in s.cast if not c.enters_at]
    got = writer.ask(
        Briefing, "You write the short briefing a player hears before stepping into a scene. Reply with a single JSON object.",
        f"Scene: {s.title}\nWhere: {s.setting}\nThe player is: {s.user_role}\n"
        f"What the player already knows: {'; '.join(s.user_knows) or 'nothing beyond the above'}\n"
        f"In front of them when it starts: {'; '.join(f'{c.name}, {c.role}' for c in here)}\n"
        f"What the player wants: {s.user_goal}\n\n"
        f"Write 3 or 4 short sentences in {VOICE.get(ui, VOICE['en'])}, speaking to the player as 'you':\n"
        "1. where they are and who they are here;\n2. what has been going on, from what the player knows;\n"
        "3. who is in front of them (name and who they are);\n4. what they want to get out of this conversation.\n"
        "One item per sentence: never join two of them in one sentence. Every sentence in the same language.\n"
        "Whenever you mention someone who isn't in front of them, say who that person is to the player or to the person in front "
        "of them (e.g. 'his best friend of eight years'), never just 'the friend' or 'she'.\n"
        "Full, easy sentences, each under about 80 characters, with normal grammar and punctuation (apostrophes as in "
        "\"the captain's\" are fine). No lists, no bullet marks, no stage directions, no quotation marks around words, "
        "no hints about how to win, and nothing the player doesn't know: never anyone's secret or private goal.\n"
        'Return {"lines": ["...", "...", "..."]}')
    end = "。" if ui == "ja" else "."
    return [ln if ln[-1] in ".!?。！？…" else ln + end for ln in (x.strip() for x in got.lines) if ln][:4]
