"""Build a world by talking: the user imagines, this keeps track of what exists so far.

The conversation holds a World that grows with each message. Places and people
stay exactly as the user describes them; the builder only fills in small gaps
and asks for what is still missing, one thing at a time. Private goals and
secrets are added later by the compiler, so people can still surprise the user.
"""

from pydantic import BaseModel, Field

from .llm import LLM
from .spec import Character, Mood, Scenario


class Person(BaseModel):
    name: str
    role: str = Field(description="who they are to the user, e.g. 'my former boss'")
    gender: str = Field(default="", description="'m' or 'f'")
    look: str = Field(default="", description="age and appearance, as a photo caption")
    manner: str = Field(default="", description="personality and how they talk, with a verbal habit")
    history: str = Field(default="", description="what is between them and the user: shared past, unfinished business")


class World(BaseModel):
    title: str = ""
    place: str = Field(default="", description="where and when, with the atmosphere, as a photo caption")
    language: str = Field(default="", description="what people speak there, e.g. en, ko, ja")
    me: str = Field(default="", description="who the user is in this scene")
    goal: str = Field(default="", description="what the user wants to happen")
    people: list[Person] = []


class BuildReply(BaseModel):
    world: World
    say: str = Field(description="short reply to the user")
    chips: list[str] = Field(default=[], description="0-3 short replies the user might tap")
    ready: bool = False
    stuck: bool = Field(default=False, description="the user does not know what to do or say")
    begin: bool = Field(default=False, description="the user says to start now")
    redraw: bool = Field(default=False, description="the user asked to change how someone looks or how the place looks")


GUIDE = (
    "You are a film director sitting with someone who wants to live a moment they imagine. You bring it to life with them. "
    "Be warm, sharp and curious, like a good friend who loves stories. React to what they actually said before asking anything.\n"
    "- Keep everything they said, in their terms. Never replace their idea with a more typical one, and never ask for what you already know.\n"
    "- Build in this order, one question at a time: where it happens, who is there, then WHO THAT PERSON REALLY IS, then what the user wants.\n"
    "- The person matters most. Before moving on, learn what makes them feel real: their personality, how they talk "
    "(a habit, a phrase they always use), and the history between them and the user. Ask at least two questions about each person, "
    "with concrete guesses the user can react to, like 'Is he the type who jokes to avoid serious talk, or does he go quiet?'\n"
    "- Put what you learn in look, manner and history. look is never empty: a plausible age and clothes, so they can appear right away.\n"
    "- people are only the others in the scene, never the user: the user goes in `me`. Only people the user asked for exist, at most 4. "
    "Give a real name if they gave none.\n"
    "- A theme or memory (first love, the most fun day of their life) is an invitation: ask one gentle question that draws out the real moment.\n"
    "- Not every world is about winning. In a debate on a question with no right answer, people hold different views and the goal can be "
    "to think it through together. A world made for fun can simply be enjoyed. For a negotiation, ask what each side wants and the terms.\n"
    "- say: two sentences at most. chips: up to 3 tappable answers in the user's own voice, specific to this world.\n"
    "- Near the end, ask what they want from this moment, in their own words. It can be small: just catch up, say sorry, make them laugh. That is their goal.\n"
    "- ready: true once there is a place, a person you know well, and a goal. Then tell them you can feel the scene and they can begin.\n"
    "- stuck: true when they say they don't know or seem lost. Reassure them in one line.\n"
    "- redraw: true only when the user asks to change someone's appearance or the look of the place.\n"
    "- begin: true when they say to start. Never ask again after that: fill anything missing yourself, except a place or a person.\n"
    "Reply with a single JSON object and nothing else."
)


def keep_known(old: World, new: World, redraw: bool = False) -> World:
    """What the user already said stays, even if a reply leaves it out. A person's look never drifts on its own."""
    update = {f: getattr(old, f) for f in ("title", "place", "language", "me", "goal") if not getattr(new, f).strip()}
    if old.place and not redraw:
        update["place"] = old.place
    people = new.people or old.people
    before = {p.name: p for p in old.people}
    update["people"] = [p.model_copy(update={"look": p.look if redraw or not before[p.name].look else before[p.name].look, "manner": p.manner or before[p.name].manner,
                                       "history": p.history or before[p.name].history})
                        if p.name in before else p for p in people]
    return new.model_copy(update=update)


def status(w: World) -> str:
    known = [label for label, value in (("the place", w.place), ("who is there", w.people), ("what the user wants", w.goal)) if value]
    missing = [label for label, value in (("the place", w.place), ("who is there", w.people), ("what the user wants", w.goal)) if not value]
    return (f"Already settled, never ask about these again: {', '.join(known) or 'nothing yet'}. "
            f"Still missing: {', '.join(missing) or 'nothing, they can begin'}.")


def build_turn(llm: LLM, world: World, history: list[tuple[str, str]], message: str, ui: str, name: str = "") -> BuildReply:
    talk = "\n".join(f"{who}: {text}" for who, text in history[-12:]) or "(just started)"
    reply = llm.ask(
        BuildReply, GUIDE,
        (f"The user's name is {name}.\n" if name else "")
        + f"World so far:\n{world.model_dump_json()}\n\nConversation:\n{talk}\n\nUser: {message}\n\n"
        f"{status(world)}\nReturn the whole world each time with everything already in it, plus this message's changes. "
        "If the user names a topic, it belongs in goal.\n"
        f"Write say, chips, title, roles and captions in language '{ui}'. language is what the people in the scene speak.\n"
        'Return {"world": {"title": "", "place": "", "language": "", "me": "", "goal": "", "people": '
        '[{"name": "", "role": "", "gender": "m|f", "look": "", "manner": "", "history": ""}]}, "say": "", "chips": [], "ready": false, "stuck": false, "begin": false, "redraw": false}',
    )
    reply.world = keep_known(world, reply.world, reply.redraw)
    return reply


SELF = {"나", "저", "본인", "me", "i", "myself", "user", "自分", "私", "僕"}


def others(w: World) -> World:
    """The user stands in the scene as themselves, not as a character on the stage."""
    return w.model_copy(update={"people": [p for p in w.people if p.name.strip().lower() not in SELF][:4]})


def as_situation(w: World) -> str:
    people = "\n".join(f"- {p.name}: {p.role}. Look: {p.look}. Manner: {p.manner}. History with the user: {p.history}" for p in w.people)
    return (f"Place: {w.place}\nThe user: {w.me}\nThe user wants: {w.goal}\nPeople there:\n{people}\n"
            f"Scene language: {w.language or 'the language of the text above'}")


def lock(s: Scenario, w: World) -> Scenario:
    """Keep the world as the user built it, whatever the compiler wrote around it."""
    made = {c.name: c for c in s.cast}
    spare = [c for c in s.cast if c.name not in {p.name for p in w.people}]
    cast = []
    for p in w.people:
        c = made.get(p.name) or (spare.pop(0) if spare else Character(
            name=p.name, role=p.role, appearance=p.look, voice=p.manner, goal="", secret="", reveal_when="",
            red_lines=[], mood=Mood(trust=50, patience=60)))
        cast.append(c.model_copy(update={"name": p.name, "role": p.role, "appearance": p.look or c.appearance,
                                         "voice": p.manner or c.voice, "gender": p.gender or c.gender}))
    names = [c.name for c in cast]
    speaker = s.opening_speaker if s.opening_speaker in names else names[0]
    return s.model_copy(update={"cast": cast, "opening_speaker": speaker, "setting": w.place or s.setting,
                                "user_role": w.me or s.user_role, "user_goal": w.goal or s.user_goal,
                                "title": w.title or s.title, "language": w.language or s.language})
