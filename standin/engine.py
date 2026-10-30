"""The scene loop: user speaks -> director picks who answers -> each character
thinks privately, then speaks.

- Every character only sees the public transcript plus its own past thoughts.
- Mood is owned by code: the model proposes a change, code clamps and applies it.
- GATE 2 (leak guard): until a secret is out, every line is checked; a line that
  gives the secret away before `reveal_when` is met is rewritten.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .llm import LLM
from .spec import Character, Expression, Mood, Scenario


class Cue(BaseModel):
    name: str
    note: str = Field(description="short stage direction for this character's next line")


class Direction(BaseModel):
    speakers: list[Cue] = Field(description="1-2 characters who respond now, in order")
    end: bool
    end_reason: str = ""


class MoodDelta(BaseModel):
    trust: int = Field(ge=-25, le=25)
    patience: int = Field(ge=-25, le=25)


class Beat(BaseModel):
    inner: str = Field(description="private thought before speaking, 1-2 sentences")
    mood_delta: MoodDelta
    expression: Expression
    say: str


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
        """Transcript as `viewer` sees it: everyone's words, only their own thoughts."""
        out = []
        for ln in self.lines:
            who = {"user": f"{self.scenario.user_role} (USER)", "narration": "*"}.get(ln.speaker, ln.speaker)
            out.append(f"[{ln.turn}] {who}: {ln.say}")
            if ln.cues:
                out.append(f"    (observed: {ln.cues})")
            if viewer and ln.speaker == viewer and ln.inner:
                out.append(f"    (you thought: {ln.inner})")
        return "\n".join(out)


def _cast_brief(c: Character, mood: Mood) -> str:
    return (f"{c.name} ({c.role}) mood trust={mood.trust} patience={mood.patience}\n"
            f"  goal: {c.goal}\n  secret: {c.secret}\n  reveal_when: {c.reveal_when}")


def direct(director: LLM, scene: Scene, turn: int) -> Direction:
    s = scene.scenario
    names = [c.name for c in s.cast if c.name not in scene.left]
    d = director.ask(
        Direction,
        "You are the director of a live rehearsal scene. You keep it realistic and tense, never theatrical. "
        "Reply with a single JSON object.",
        f"Setting: {s.setting}\nUser plays: {s.user_role}; user's goal: {s.user_goal}\n"
        f"Phases: {' -> '.join(s.phases)}\nThis is user turn {turn} of at most {s.max_turns}.\n\n"
        "Characters (private info visible only to you):\n"
        + "\n".join(_cast_brief(c, scene.moods[c.name]) for c in s.cast if c.name in names)
        + f"\n\nTranscript so far:\n{scene.public()}\n\n"
        "Decide who reacts to the user's last line. Usually one person; two only if a second one would really "
        "jump in. A character with patience near 0 wants to wrap up. Give each a short stage direction "
        "(what to push on, whether to interrupt, stay silent-ish, change tone). "
        "Set end=true only if the conversation has reached a natural conclusion or someone walks out.\n"
        f'Names: {names}. Return {{"speakers": [{{"name": "...", "note": "..."}}], "end": false, "end_reason": ""}}',
    )
    d.speakers = [c for c in d.speakers if c.name in names][:2]
    if not d.speakers:  # never stall the scene on a bad name
        d.speakers = [Cue(name=names[0], note="respond to the user")]
    return d


def act(actor: LLM, scene: Scene, c: Character, note: str, extra: str = "") -> Beat:
    s, mood = scene.scenario, scene.moods[c.name]
    others = "\n".join(f"- {o.name}: {o.role}" for o in s.cast if o.name != c.name) or "- (no one else)"
    secret_rule = ("You have already let your secret out; you may refer to it." if c.name in scene.revealed else
                   f"Keep your secret unless the user has clearly done this: {c.reveal_when}. Do not hint at it "
                   "before then.")
    return actor.ask(
        Beat,
        f"You are {c.name}, a real person, not an assistant. Stay fully in character. Speak in language "
        f"'{s.language}'. Reply with a single JSON object.",
        f"Scene: {s.setting}\nYou are: {c.role}. Look: {c.appearance}. How you talk: {c.voice}\n"
        f"Your private goal: {c.goal}\nYour secret: {c.secret}\n{secret_rule}\n"
        f"You will not: {'; '.join(c.red_lines)}\n"
        f"The person in front of you: {s.user_role}.\nOthers present:\n{others}\n\n"
        f"Your mood now: trust {mood.trust}/100, patience {mood.patience}/100. Low trust = guarded, skeptical; "
        "low patience = short, cutting, wants to end. React to substance: vague claims and repetition cost "
        "trust and patience; concrete evidence, good questions and respect earn trust. Lines marked "
        "'observed' are what you could see and hear while they spoke (pauses, eye contact, fidgeting); react "
        "to them only when a real person in your role would notice and care.\n\n"
        f"Transcript (with your own past thoughts):\n{scene.public(viewer=c.name)}\n\n"
        f"Director's note for your next line: {note}\n{extra}\n"
        "Write your private thought first, then what you actually say out loud: 1-3 sentences, natural spoken "
        "language, no stage directions, no narration.\n"
        'Return {"inner": "...", "mood_delta": {"trust": -25..25, "patience": -25..25}, '
        '"expression": "neutral|pleased|skeptical|annoyed", "say": "..."}',
    )


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


def respond(actor: LLM, checker: LLM, scene: Scene, turn: int, c: Character, note: str) -> Line:
    beat = act(actor, scene, c, note)
    gate = ""
    if c.name not in scene.revealed:
        chk = leak_check(checker, scene, c, beat.say)
        chk.reveals = chk.reveals and bool(chk.quote.strip()) and chk.quote.strip() in beat.say  # no quote, no leak
        if chk.reveals and not chk.earned:
            gate = f"rewritten: early leak ({chk.why})"
            beat = act(actor, scene, c, note, extra=(
                f"IMPORTANT: your draft line \"{beat.say}\" gave away your secret too early ({chk.why}). "
                "Say something else that keeps it hidden."))
        elif chk.reveals:
            scene.revealed.add(c.name)
            gate = f"secret revealed, earned ({chk.why})"
    m = scene.moods[c.name]
    m.trust = max(0, min(100, m.trust + beat.mood_delta.trust))
    m.patience = max(0, min(100, m.patience + beat.mood_delta.patience))
    return Line(turn, c.name, beat.say, beat.inner, beat.expression, m.model_dump(), gate)


LEAVES = {"ko": "{name}이(가) 더 듣지 않고 자리를 뜬다.", "ja": "{name}は話を切り上げて席を立つ。"}


def open_scene(scenario: Scenario) -> Scene:
    scene = Scene(scenario)
    s = scenario
    scene.lines.append(Line(0, s.opening_speaker, s.opening_line, expression="neutral",
                            mood=scene.moods[s.opening_speaker].model_dump()))
    return scene


def step(scene: Scene, turn: int, said: str, director: LLM, actor: LLM, checker: LLM,
         cues: str = "") -> list[Line]:
    """One user turn. Returns the new lines (the user's first); sets scene.ended when the scene is over."""
    s = scene.scenario
    new: list[Line] = []

    def add(ln: Line) -> None:
        scene.lines.append(ln)
        new.append(ln)

    add(Line(turn, "user", said, cues=cues))
    for e in s.events:
        if e.at_turn == turn:
            add(Line(turn, "narration", e.description))
    d = direct(director, scene, turn)
    for cue in d.speakers:
        add(respond(actor, checker, scene, turn, s.character(cue.name), cue.note))
        if scene.moods[cue.name].patience == 0:  # out of patience: that was their last word
            scene.left.add(cue.name)
            leave = LEAVES.get(s.language, "{name} has had enough and leaves the conversation.")
            add(Line(turn, "narration", leave.format(name=cue.name)))
    if len(scene.left) == len(s.cast):
        scene.ended = "everyone ran out of patience"
    elif d.end:
        scene.ended = d.end_reason or "director ended the scene"
    elif turn >= s.max_turns:
        scene.ended = "time is up"
    return new


def run_scene(scenario: Scenario, director: LLM, actor: LLM, checker: LLM,
              user_turn: Callable[[Scene, int], str], show: Callable[[Line], None]) -> Scene:
    scene = open_scene(scenario)
    show(scene.lines[0])
    for turn in range(1, scenario.max_turns + 1):
        said = user_turn(scene, turn).strip()
        if said in ("/end", ""):
            scene.ended = "user ended the scene"
            break
        for ln in step(scene, turn, said, director, actor, checker)[1:]:
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
