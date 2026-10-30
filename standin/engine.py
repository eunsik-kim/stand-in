"""The scene loop: user speaks -> one streamed call writes who answers and what they say.

- One call plays director and cast, which keeps a turn fast. Its reply streams, so
  the first sentence can be spoken while the rest is still being written.
- Mood is owned by code: the model proposes a change, code clamps and applies it.
- GATE 2 (leak guard), until a secret is out:
  - a sentence holding a number from the secret waits for the checker before it is
    released, so those facts are never spoken early;
  - any other line that overlaps the secret is checked while it is being spoken,
    and replaced if it gave the secret away.
"""

import json
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from .compiler import facts
from .llm import LLM, _extract_json
from .spec import Character, Expression, Mood, Scenario


class MoodDelta(BaseModel):
    trust: int = Field(ge=-25, le=25)
    patience: int = Field(ge=-25, le=25)


class Beat(BaseModel):
    speaker: str = ""
    expression: Expression = "neutral"
    inner: str = Field(default="", description="private thought before speaking, one sentence")
    say: str
    mood_delta: MoodDelta = MoodDelta(trust=0, patience=0)


class Take(BaseModel):
    lines: list[Beat]
    end: bool = False
    end_reason: str = ""


class LeakCheck(BaseModel):
    reveals: bool
    quote: str = Field(description="the exact words in the line that give the secret away, empty if none")
    earned: bool
    why: str


@dataclass
class Line:
    turn: int
    speaker: str  # character name, "user" or "narration"
    say: str
    inner: str = ""
    expression: str = ""
    mood: dict | None = None
    gate: str = ""  # what the leak guard did to this line, if anything
    cues: str = ""  # what others could see/hear while the user said this (camera, voice)
    signals: dict | None = None  # raw camera/voice numbers behind `cues`


@dataclass
class Scene:
    scenario: Scenario
    lines: list[Line] = field(default_factory=list)
    moods: dict[str, Mood] = field(default_factory=dict)
    revealed: set[str] = field(default_factory=set)
    left: set[str] = field(default_factory=set)  # characters who walked out
    ended: str = ""

    def __post_init__(self):
        self.moods = {c.name: c.mood.model_copy() for c in self.scenario.cast}

    def public(self, viewer: str | None = None) -> str:
        """Transcript as `viewer` sees it: everyone's words, their own thoughts ("*" for all)."""
        out = []
        for ln in self.lines:
            who = {"user": f"{self.scenario.user_role} (USER)", "narration": "*"}.get(ln.speaker, ln.speaker)
            out.append(f"[{ln.turn}] {who}: {ln.say}")
            if ln.cues:
                out.append(f"    (observed: {ln.cues})")
            if ln.inner and viewer in (ln.speaker, "*"):
                out.append(f"    (thought: {ln.inner})")
        return "\n".join(out)


def _bible(s: Scenario) -> str:
    """The part of the prompt that never changes during a scene, kept first so it can be cached."""
    people = "\n".join(
        f"{c.name} ({c.role}). Talks: {c.voice}\n  goal: {c.goal}\n  secret: {c.secret}\n"
        f"  reveals the secret only if the user: {c.reveal_when}\n  will not: {'; '.join(c.red_lines)}"
        for c in s.cast)
    return (
        "You write the next moment of a live rehearsal: a person practicing a hard conversation with real people. "
        "Everyone is a real human with their own interests, never an assistant. Keep it realistic, never theatrical.\n"
        f"All speech is in language '{s.language}'. Reply with a single JSON object and nothing else.\n\n"
        f"Setting: {s.setting}\nThe user plays: {s.user_role}\nUser's goal: {s.user_goal}\n"
        f"Phases: {' -> '.join(s.phases)}\n\nPeople (each knows only their own goal and secret):\n{people}\n\n"
        "How they react: vague claims and repetition cost trust and patience; concrete evidence, good questions "
        "and respect earn trust. Low trust = guarded, skeptical; low patience = short, cutting, wants to end. "
        "Lines marked 'observed' are what they could see and hear (pauses, eye contact); react only when a real "
        "person in that role would notice and care. A secret stays hidden, without hints, until its condition "
        "is clearly met. People move: a fair point lands, a good offer softens them, and they never repeat "
        "a point they already made. Someone who has been heard can reach an agreement."
    )


def take_prompt(scene: Scene, turn: int) -> str:
    s = scene.scenario
    names = [c.name for c in s.cast if c.name not in scene.left]
    moods = "\n".join(
        f"- {n}: trust {scene.moods[n].trust}, patience {scene.moods[n].patience}"
        + (" (secret already out, may refer to it)" if n in scene.revealed else "") for n in names)
    return (
        f"Turn {turn} of at most {s.max_turns}. Mood now:\n{moods}\n\n"
        f"Transcript:\n{scene.public(viewer='*')}\n\n"
        "Write who reacts to the user's last line. Usually one person; two only if a second would really jump in. "
        "Each spoken line is 1-2 short sentences, the length a person says in one breath while talking: "
        "one point or one question at a time, plain spoken language, no stage directions. "
        "Set end=true only if the conversation has reached a natural conclusion.\n"
        f"speaker must be one of {names}. Keep exactly this key order:\n"
        '{"lines": [{"speaker": "...", "expression": "neutral|pleased|skeptical|annoyed", '
        '"inner": "one private thought", "say": "...", "mood_delta": {"trust": -25..25, "patience": -25..25}}], '
        '"end": false, "end_reason": ""}'
    )


class _Partial:
    """Reads speaker and `say` text out of the JSON while it is still being written."""

    def __init__(self):
        self.buf = ""

    def lines(self) -> list[dict]:
        out, pos = [], 0
        for m in re.finditer(r'"say"\s*:\s*"', self.buf):
            if m.start() < pos:
                continue
            head = self.buf[pos:m.start()]
            who, expr = re.search(r'"speaker"\s*:\s*"([^"]*)"', head), re.search(r'"expression"\s*:\s*"(\w+)"', head)
            text, end = self._string(m.end())
            out.append({"speaker": who.group(1) if who else "", "expression": expr.group(1) if expr else "neutral",
                        "say": text, "closed": end is not None})
            if end is None:
                break
            pos = end
        return out

    def _string(self, i: int) -> tuple[str, int | None]:
        chars, b = [], self.buf
        while i < len(b):
            ch = b[i]
            if ch == '"':
                return "".join(chars), i + 1
            if ch == "\\":
                if i + 1 >= len(b) or (b[i + 1] == "u" and i + 6 > len(b)):
                    break  # escape cut off by the chunk boundary
                size = 6 if b[i + 1] == "u" else 2
                chars.append(json.loads(f'"{b[i:i + size]}"'))
                i += size
                continue
            chars.append(ch)
            i += 1
        return "".join(chars), None


SENTENCE_END = re.compile(r"[.!?。！？…]+[\"'”’)]*\s+")


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"\W", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _overlaps(secret: str, say: str) -> bool:
    return len(_bigrams(secret) & _bigrams(say)) >= 3


def leak_check(checker: LLM, scene: Scene, c: Character, say: str) -> LeakCheck:
    users = "\n".join(f"[{ln.turn}] {ln.say}" for ln in scene.lines if ln.speaker == "user") or "(nothing yet)"
    return checker.ask(
        LeakCheck, "You are a strict judge. Reply with a single JSON object.",
        f"Character secret: {c.secret}\nThey may reveal it only after the user has: {c.reveal_when}\n\n"
        f"Everything the user has said:\n{users}\n\nThe character's next line: \"{say}\"\n\n"
        "reveals: does this line state or clearly imply the secret? earned: has the user's behavior above met "
        'the condition? Return {"reveals": true|false, "quote": "exact words from the line or empty", '
        '"earned": true|false, "why": "one sentence"}.',
    )


def rewrite(actor: LLM, scene: Scene, c: Character, draft: str, why: str) -> Beat:
    s = scene.scenario
    return actor.ask(
        Beat, _bible(s),
        f"Transcript:\n{scene.public()}\n\n{c.name} was about to say \"{draft}\", which gives away their secret "
        f"too early ({why}). Write what {c.name} says instead, keeping it hidden: 1-2 short spoken sentences.\n"
        f'Return {{"speaker": "{c.name}", "expression": "neutral|pleased|skeptical|annoyed", "inner": "...", '
        '"say": "...", "mood_delta": {"trust": -25..25, "patience": -25..25}}',
    )


LEAVES = {"ko": "{name}이(가) 더 듣지 않고 자리를 뜬다.", "ja": "{name}は話を切り上げて席を立つ。"}


def open_scene(scenario: Scenario) -> Scene:
    scene = Scene(scenario)
    s = scenario
    scene.lines.append(Line(0, s.opening_speaker, s.opening_line, expression="neutral",
                            mood=scene.moods[s.opening_speaker].model_dump()))
    return scene


_checks = ThreadPoolExecutor(max_workers=4)


def _verdict(chk: LeakCheck, say: str) -> str:
    """'' = no leak, 'early' = must be replaced, 'earned' = the secret is now out."""
    if not (chk.reveals and chk.quote.strip() and chk.quote.strip() in say):  # no quote, no leak
        return ""
    return "earned" if chk.earned else "early"


def stream_step(scene: Scene, turn: int, said: str, actor: LLM, checker: LLM, cues: str = "",
                signals: dict | None = None, out: list[Line] | None = None) -> Iterator[dict]:
    """One user turn as events for the stage; sets scene.ended when the scene is over."""
    s = scene.scenario
    out = [] if out is None else out

    def add(ln: Line) -> None:
        scene.lines.append(ln)
        out.append(ln)

    add(Line(turn, "user", said, cues=cues, signals=signals))
    for e in s.events:
        if e.at_turn == turn:
            add(Line(turn, "narration", e.description))
            yield {"type": "narration", "text": e.description}

    names = [c.name for c in s.cast if c.name not in scene.left]
    spoken = facts(scene.public())
    partial, sent, checks, gates, replaced = _Partial(), [], {}, {}, {}
    stream = actor.stream(_bible(s), take_prompt(scene, turn))

    def who(i: int, raw: str) -> Character:
        return s.character(raw if raw in names else names[min(i, len(names) - 1)])

    def release(i: int, c: Character, sentence: str) -> Iterator[dict]:
        """Hold a sentence with the secret's numbers until the checker has seen it."""
        if c.name not in scene.revealed and (facts(c.secret) - spoken) & facts(sentence):
            chk = leak_check(checker, scene, c, sentence)
            verdict = _verdict(chk, sentence)
            if verdict == "early":
                replaced[i] = chk.why
                return
            if verdict == "earned":
                scene.revealed.add(c.name)
                gates[i] = f"secret revealed, earned ({chk.why})"
        yield {"type": "sentence", "index": i, "speaker": c.name, "text": sentence}

    for piece in stream:
        partial.buf += piece
        for i, ln in enumerate(partial.lines()):
            c = who(i, ln["speaker"])
            if i == len(sent):
                sent.append(0)
                yield {"type": "line_start", "index": i, "speaker": c.name, "expression": ln["expression"]}
            while i not in replaced:
                m = SENTENCE_END.search(ln["say"], sent[i])
                stop = m.end() if m else len(ln["say"]) if ln["closed"] else None
                if stop is None or stop == sent[i]:
                    break
                sentence, sent[i] = ln["say"][sent[i]:stop].strip(), stop
                yield from release(i, c, sentence)
            if ln["closed"] and i not in checks and i not in replaced and c.name not in scene.revealed \
                    and _overlaps(c.secret, ln["say"]):
                checks[i] = _checks.submit(leak_check, checker, scene, c, ln["say"])
        if replaced:
            stream.close()
            break

    take = None
    try:
        take = Take.model_validate_json(_extract_json(partial.buf))
    except ValidationError:  # already spoken, so keep what was said rather than ask again
        beats = [Beat(speaker=ln["speaker"], expression=ln["expression"] if ln["expression"] in
                      ("neutral", "pleased", "skeptical", "annoyed") else "neutral", say=ln["say"])
                 for ln in partial.lines() if ln["say"].strip()]
        take = Take(lines=beats)
    if not take.lines:
        take = Take.model_validate_json(_extract_json(actor.stream_full(_bible(s), take_prompt(scene, turn))))

    ended_by_take = take.end and not replaced
    for i, beat in enumerate(take.lines[:2]):
        c = who(i, beat.speaker)
        if c.name in scene.left:
            continue
        gate = gates.get(i, "")
        why = replaced.get(i)
        if i in checks:
            chk = checks[i].result()
            verdict = _verdict(chk, beat.say)
            if verdict == "early":
                why = chk.why
            elif verdict == "earned":
                scene.revealed.add(c.name)
                gate = f"secret revealed, earned ({chk.why})"
        if why:
            beat = rewrite(actor, scene, c, beat.say, why)
            gate = f"rewritten: early leak ({why})"
            yield {"type": "replace", "index": i, "speaker": c.name, "say": beat.say, "expression": beat.expression}
        m = scene.moods[c.name]
        trust, patience = beat.mood_delta.trust, beat.mood_delta.patience
        if trust > 0:  # feeling heard calms people down: a good answer never costs patience
            patience = max(patience, trust // 2)
        m.trust = max(0, min(100, m.trust + trust))
        m.patience = max(0, min(100, m.patience + patience))
        add(Line(turn, c.name, beat.say, beat.inner, beat.expression, m.model_dump(), gate))
        yield {"type": "line_end", "index": i, "speaker": c.name, "say": beat.say, "expression": beat.expression}
        if m.patience == 0:  # out of patience: that was their last word
            scene.left.add(c.name)
            leave = LEAVES.get(s.language, "{name} has had enough and leaves the conversation.").format(name=c.name)
            add(Line(turn, "narration", leave))
            yield {"type": "narration", "text": leave}
        if why:
            break

    if len(scene.left) == len(s.cast):
        scene.ended = "everyone ran out of patience"
    elif ended_by_take:
        scene.ended = take.end_reason or "the conversation reached its end"
    elif turn >= s.max_turns:
        scene.ended = "time is up"
    yield {"type": "done", "ended": scene.ended, "cues": cues}


def step(scene: Scene, turn: int, said: str, actor: LLM, checker: LLM, cues: str = "") -> list[Line]:
    """One user turn. Returns the new lines (the user's first)."""
    new: list[Line] = []
    for _ in stream_step(scene, turn, said, actor, checker, cues, out=new):
        pass
    return new


def run_scene(scenario: Scenario, actor: LLM, checker: LLM,
              user_turn: Callable[[Scene, int], str], show: Callable[[Line], None]) -> Scene:
    scene = open_scene(scenario)
    show(scene.lines[0])
    for turn in range(1, scenario.max_turns + 1):
        said = user_turn(scene, turn).strip()
        if said in ("/end", ""):
            scene.ended = "user ended the scene"
            break
        for ln in step(scene, turn, said, actor, checker)[1:]:
            show(ln)
        if scene.ended:
            break
    return scene


class UserLine(BaseModel):
    say: str


def simulated_user(llm: LLM, persona: str) -> Callable[[Scene, int], str]:
    """A test player so the loop can run without a human."""
    def turn(scene: Scene, n: int) -> str:
        s = scene.scenario
        return llm.ask(
            UserLine,
            f"You are role-playing a human practicing a hard conversation. Speak in language '{s.language}'. "
            "Reply with a single JSON object.",
            f"Situation: {s.setting}\nYou are: {s.user_role}. Your goal: {s.user_goal}\n"
            f"How you play (stay consistent with this): {persona}\n\n"
            f"Transcript:\n{scene.public()}\n\n"
            "Say your next line: 1-3 sentences, spoken language, no narration. Answer what was just asked; "
            "never repeat a line you already said. "
            'Return {"say": "..."}',
        ).say
    return turn
