"""The scene loop: user speaks -> one streamed call writes who answers and what they say.

- One call plays director and cast, which keeps a turn fast. Its reply streams, so
  the first sentence can be spoken while the rest is still being written.
- Mood is owned by code: the model proposes a change, code clamps and applies it.
- Secrets: the same call that writes a line says whose secret it gave up, which also
  catches paraphrase and third parties. What is on screen has been said; nothing is taken back.
- The scene knows who is quiet, who was spoken to and what just happened, and the
  ending follows from what the player actually earned.
"""

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ValidationError

from .llm import LLM, _extract_json
from .spec import SPOKEN, Character, Event, Expression, Mood, Scenario


class MoodDelta(BaseModel):
    trust: int = Field(ge=-25, le=25)
    patience: int = Field(ge=-25, le=25)


class Beat(BaseModel):
    speaker: str = ""
    expression: Expression = "neutral"
    inner: str = Field(default="", description="private thought before speaking, one sentence")
    gesture: str = ""  # what the listener can see them do as they speak
    say: str
    reveals: str = ""  # whose secret this line just gave up, if anyone's
    mood_delta: MoodDelta = MoodDelta(trust=0, patience=0)


class Take(BaseModel):
    lines: list[Beat]
    advance: bool = False  # this moment has peaked: bring on the next turning point
    end: bool = False
    end_reason: str = ""
    closing: str = ""  # on the last turn: one line of narration showing how it ends


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
    auto: bool = False  # the stand-in said this, not the player


@dataclass
class Scene:
    scenario: Scenario
    lines: list[Line] = field(default_factory=list)
    moods: dict[str, Mood] = field(default_factory=dict)
    revealed: set[str] = field(default_factory=set)
    left: set[str] = field(default_factory=set)  # characters who walked out
    arrived: set[str] = field(default_factory=set)  # people who came in during the scene
    beat: int = 0  # how many planned turning points have happened
    advance: bool = False  # the scene judged that the next one should land now
    hinted: set[str] = field(default_factory=set)  # who has let their one unprompted clue slip
    ended: str = ""

    def __post_init__(self):
        self.moods = {c.name: c.mood.model_copy() for c in self.scenario.cast}

    def public(self, viewer: str | None = None) -> str:
        """Transcript as `viewer` sees it: everyone's words, their own thoughts ("*" for all)."""
        out = []
        for ln in self.lines:
            who = {"user": f"{self.scenario.user_role} (USER)", "narration": "*"}.get(ln.speaker, ln.speaker)
            if viewer == "record" and ln.auto:
                who += " (auto)"
            out.append(f"[{ln.turn}] {who}: {ln.say}")
            if ln.cues:
                out.append(f"    (observed: {ln.cues})")
            if ln.inner and viewer in (ln.speaker, "*"):
                out.append(f"    (thought: {ln.inner})")
        return "\n".join(out)


def _bible(s: Scenario) -> str:
    """The part of the prompt that never changes during a scene, kept first so it can be cached."""
    people = "\n".join(
        f"{c.name} ({c.role}). Talks: {c.voice}\n  already happened with the user: {c.history}\n  goal: {c.goal}\n  secret: {c.secret}\n"
        f"  reveals the secret only if the user: {c.reveal_when}\n  will not: {'; '.join(c.red_lines)}"
        for c in s.cast)
    return (
        "You write the next moment of a live rehearsal: a person practicing a hard conversation with real people. "
        "Everyone is a real human with their own interests, never an assistant. Keep it realistic, never theatrical.\n"
        f"All speech is in language '{s.language}'. Reply with a single JSON object and nothing else.\n\n"
        f"Setting: {s.setting}\nThe user plays: {s.user_role}\nUser's goal: {s.user_goal}\n"
        + (f"The user already knows: {'; '.join(s.user_knows)}\n" if s.user_knows else "")
        + f"Phases: {' -> '.join(s.phases)}\n\nPeople (each knows only their own goal and secret):\n{people}\n\n"
        "How they react: vague claims and repetition cost trust and patience; concrete evidence, good questions "
        "and respect earn trust. Low trust = guarded, skeptical; low patience = short, cutting, wants to end. "
        "Lines marked 'observed' are what they could see and hear (pauses, eye contact); react only when a real "
        "person in that role would notice and care. A secret stays hidden until its condition is clearly met, or until the "
        "user confronts them with a specific fact that makes hiding it impossible. Nobody tells a secret in front of the person it "
        "concerns unless forced to. Nobody tells the user what to ask or do next. "
        f"{SPOKEN} People move: a fair point lands, a good offer softens them, and they never repeat "
        "a point they already made. Someone who has been heard can reach an agreement."
    )


CLUE_TRUST, OPEN_TRUST = 50, 70


def _openness(scene: Scene, name: str) -> str:
    """Code decides when someone starts to open up, so a good conversation always leads somewhere."""
    trust = scene.moods[name].trust
    if trust >= OPEN_TRUST:
        return " (trusts the user now: if the user asks anything near their secret, they tell it plainly)"
    if trust >= CLUE_TRUST and name not in scene.hinted:
        scene.hinted.add(name)  # once: the second tease of the same thing shows the machinery
        return (" (warming up: this once, let a small clue slip that something is behind their position, "
                "without stating the secret itself)")
    return ""


def present(scene: Scene, turn: int) -> list[str]:
    return [c.name for c in scene.scenario.cast
            if (not c.enters_at or c.name in scene.arrived) and c.name not in scene.left]


def next_beat(scene: Scene) -> Event | None:
    beats = sorted(scene.scenario.events, key=lambda e: e.at_turn)
    return beats[scene.beat] if scene.beat < len(beats) else None


def _who_must_speak(scene: Scene, names: list[str], fresh: list[str]) -> str:
    said = next((ln.say for ln in reversed(scene.lines) if ln.speaker == "user"), "")
    called = [n for n in names if n in said or n.split()[-1] in said]
    last = {n: max([ln.turn for ln in scene.lines if ln.speaker == n], default=-1) for n in names}
    turn = max((ln.turn for ln in scene.lines), default=0)
    quiet = [n for n in names if n not in called and n not in fresh and turn - last[n] >= 2]
    return "".join([
        f"The user spoke to {', '.join(called)}: they answer, nobody answers for them.\n" if called else "",
        f"{', '.join(fresh)} just came in and must speak now.\n" if fresh else "",
        f"{', '.join(quiet)} have been quiet: bring one in if they would really have something to say.\n" if quiet else ""])


def _ending_note(scene: Scene, turn: int) -> str:
    s = scene.scenario
    if turn == s.max_turns - 1:
        return "The next turn is the last. Bring the story to its peak now, and leave the user something to answer.\n"
    if turn < s.max_turns:
        return "End the scene early (end=true) as soon as a verdict is given, a deal is signed, someone leaves for good, or the user closes it.\n"
    earned = bool(scene.revealed) or max((m.trust for m in scene.moods.values()), default=0) >= OPEN_TRUST
    return ("This is the last turn. Answer the user's last line, then close the story; the last line is a statement, never a question. "
            + ("The user earned this: let it land." if earned else
               "The user has not earned a win: nobody gives in and red lines hold. End on a refusal, a walk-out or a door closing.")
            + " Set end=true, and in closing write one line of stage narration showing what physically happens last.\n")


def _their_last_lines(scene: Scene, names: list[str]) -> str:
    lines = [f"{n}: " + " / ".join(ln.say for ln in [x for x in scene.lines if x.speaker == n][-2:])
             for n in names if any(x.speaker == n for x in scene.lines)]
    return "Each person's last lines; do not repeat their structure, opening or complaint:\n" + "\n".join(lines) + "\n\n" if lines else ""


def take_prompt(scene: Scene, turn: int, happened: Event | None = None) -> str:
    s = scene.scenario
    names = present(scene, turn)
    moods = "\n".join(
        f"- {n}: trust {scene.moods[n].trust}, patience {scene.moods[n].patience}"
        + (" (secret already out, may refer to it)" if n in scene.revealed else _openness(scene, n)) for n in names)
    absent = [c.name for c in s.cast if c.name not in names and c.name not in scene.left]
    coming = next_beat(scene)
    return (
        (f"This just happened: {happened.description} The next line reacts to it, and anyone it names acts it out.\n\n" if happened else "")
        + f"Turn {turn} of at most {s.max_turns}. Mood now:\n{moods}\n\n"
        + (f"Not in the room yet, so they cannot speak and nobody mentions them: {', '.join(absent)}.\n\n" if absent else "")
        + f"Transcript:\n{scene.public(viewer='*')}\n\n" + _their_last_lines(scene, names)
        + "Write who reacts to the user's last line, one or two people. "
        "Each spoken line is what a person says in one breath: one point or one question at a time, no stage directions. "
        "gesture is what the user can see them do while saying it, 2-6 words in the language of the scene: a hand stops, eyes drop, a chair "
        "pushed back. It shows how they really feel, sometimes against their words, and never repeats. Leave it empty when nothing shows.\n"
        "Every reply must move the story: a new fact, a decision, an action or a change of heart. Never reuse a phrase, joke or "
        "complaint already in the transcript, and do not echo the last words of the user's line back at them. "
        "A private alliance stays private in front of others. Set reveals to a person's name on the line where their secret "
        "actually comes out, whoever says it; otherwise leave it empty.\n"
        + _who_must_speak(scene, names, happened.enters if happened else []) + _ending_note(scene, turn)
        + (f"You are also the director. The next planned turning point is: {coming.description} Set advance=true at the moment "
           "this would hit hardest, which may be now; never let the scene drag.\n" if coming else "")
        + f"speaker must be one of {names}. Keep exactly this key order:\n"
        '{"lines": [{"speaker": "...", "expression": "neutral|pleased|skeptical|annoyed", '
        '"inner": "one private thought", "gesture": "...", "say": "...", "reveals": "", "mood_delta": {"trust": -25..25, "patience": -25..25}}], '
        '"advance": false, "end": false, "end_reason": "", "closing": ""}'
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
            body = re.search(r'"gesture"\s*:\s*"([^"]*)"', head)
            text, end = self._string(m.end())
            out.append({"speaker": who.group(1) if who else "", "expression": expr.group(1) if expr else "neutral",
                        "gesture": body.group(1) if body else "", "say": text, "closed": end is not None})
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


MOST_LINES = 3
STAGE = re.compile(r"\*[^*]*\*")  # the odd *sigh* written as stage direction
SENTENCE_END = re.compile(r"[.!?。！？…]+[\"'”’)]*\s+")


LEAVES = {"ko": "{name}{ga} 더 듣지 않고 자리를 뜬다.", "ja": "{name}は話を切り上げて席を立つ。"}


def _ga(name: str) -> str:
    last = ord(name[-1]) - 0xAC00
    return "이" if 0 <= last < 11172 and last % 28 else "가"


def open_scene(scenario: Scenario) -> Scene:
    scene = Scene(scenario)
    s = scenario
    scene.lines.append(Line(0, s.opening_speaker, s.opening_line, expression="neutral",
                            mood=scene.moods[s.opening_speaker].model_dump()))
    return scene


def stream_step(scene: Scene, turn: int, said: str, actor: LLM, cues: str = "",
                signals: dict | None = None, out: list[Line] | None = None, auto: bool = False) -> Iterator[dict]:
    """One user turn as events for the stage; sets scene.ended when the scene is over."""
    s = scene.scenario
    out = [] if out is None else out

    def add(ln: Line) -> None:
        scene.lines.append(ln)
        out.append(ln)

    add(Line(turn, "user", said, cues=cues, signals=signals, auto=auto))
    e, happened = next_beat(scene), None
    # it lands when the moment calls for it, and may wait one turn for that moment
    if e and ((scene.advance and turn >= e.at_turn - 1) or turn >= min(e.at_turn + 1, s.max_turns - 1)):
        scene.beat += 1
        scene.advance = False
        scene.arrived |= set(e.enters)
        add(Line(turn, "narration", e.description))
        yield {"type": "narration", "text": e.description, "shot": e.at_turn, "enters": e.enters}
        happened = e

    names = present(scene, turn)
    partial, sent = _Partial(), []
    prompt = take_prompt(scene, turn, happened)
    stream = actor.stream(_bible(s), prompt)

    def who(i: int, raw: str) -> Character:
        return s.character(raw if raw in names else names[min(i, len(names) - 1)])

    for piece in stream:
        partial.buf += piece
        for i, ln in enumerate(partial.lines()[:MOST_LINES]):
            c = who(i, ln["speaker"])
            if i == len(sent):
                sent.append(0)
                yield {"type": "line_start", "index": i, "speaker": c.name, "expression": ln["expression"], "gesture": ln["gesture"]}
            while True:
                m = SENTENCE_END.search(ln["say"], sent[i])
                stop = m.end() if m else len(ln["say"]) if ln["closed"] else None
                if stop is None or stop == sent[i]:
                    break
                sentence, sent[i] = STAGE.sub("", ln["say"][sent[i]:stop]).strip(), stop
                if sentence:
                    yield {"type": "sentence", "index": i, "speaker": c.name, "text": sentence}

    take = None
    try:
        take = Take.model_validate_json(_extract_json(partial.buf))
    except ValidationError:  # already spoken, so keep what was said rather than ask again
        beats = [Beat(speaker=ln["speaker"], expression=ln["expression"] if ln["expression"] in
                      ("neutral", "pleased", "skeptical", "annoyed") else "neutral", say=ln["say"])
                 for ln in partial.lines() if ln["say"].strip()]
        take = Take(lines=beats)
    if not take.lines:
        take = Take.model_validate_json(_extract_json(actor.stream_full(_bible(s), prompt)))

    ended_by_take = take.end
    scene.advance = take.advance
    for i, beat in enumerate(take.lines[:MOST_LINES]):
        c = who(i, beat.speaker)
        if c.name in scene.left:
            continue
        beat.say = STAGE.sub("", beat.say).strip()
        m = scene.moods[c.name]
        trust, patience = beat.mood_delta.trust, beat.mood_delta.patience
        if trust > 0:  # feeling heard calms people down: a good answer never costs patience
            patience = max(patience, trust // 2)
        m.trust = max(0, min(100, m.trust + trust))
        m.patience = max(0, min(100, m.patience + patience))
        add(Line(turn, c.name, beat.say, beat.inner, beat.expression, m.model_dump()))
        yield {"type": "line_end", "index": i, "speaker": c.name, "say": beat.say, "expression": beat.expression}
        owner = beat.reveals.strip()
        if owner in {x.name for x in s.cast} and owner not in scene.revealed:
            scene.revealed.add(owner)
            yield {"type": "secret", "speaker": owner}
        if m.patience == 0:  # out of patience: that was their last word
            scene.left.add(c.name)
            leave = LEAVES.get(s.language, "{name} has had enough and leaves the conversation.").format(name=c.name, ga=_ga(c.name))
            add(Line(turn, "narration", leave))
            yield {"type": "narration", "text": leave}

    if take.closing.strip():
        add(Line(turn, "narration", take.closing.strip()))
        yield {"type": "narration", "text": take.closing.strip()}
    if len(scene.left) == len(s.cast):
        scene.ended = "everyone ran out of patience"
    elif ended_by_take:
        scene.ended = take.end_reason or "the conversation reached its end"
    elif turn >= s.max_turns:
        scene.ended = "time is up"
    yield {"type": "done", "ended": scene.ended, "cues": cues, "found": len(scene.revealed)}


def step(scene: Scene, turn: int, said: str, actor: LLM, cues: str = "") -> list[Line]:
    """One user turn. Returns the new lines (the user's first)."""
    new: list[Line] = []
    for _ in stream_step(scene, turn, said, actor, cues, out=new):
        pass
    return new


def run_scene(scenario: Scenario, actor: LLM,
              user_turn: Callable[[Scene, int], str], show: Callable[[Line], None]) -> Scene:
    scene = open_scene(scenario)
    show(scene.lines[0])
    for turn in range(1, scenario.max_turns + 1):
        said = user_turn(scene, turn).strip()
        if said in ("/end", ""):
            scene.ended = "user ended the scene"
            break
        for ln in step(scene, turn, said, actor)[1:]:
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
            f"Say your next line, one breath of real speech, no narration. {SPOKEN} Answer what was just asked; "
            "never repeat a line you already said. "
            'Return {"say": "..."}',
        ).say
    return turn
