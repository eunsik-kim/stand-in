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
    manner: str = Field(default="", description="personality and how they talk")


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


GUIDE = (
    "You help someone build a rehearsal world by talking, like setting a stage together. They imagine it, you make it real.\n"
    "- Keep everything they said, in their terms. Never replace their idea with a more typical one.\n"
    "- If they offer a theme or a memory instead of a scene (first love, the most fun moment of their life), be warm and curious: "
    "ask one gentle question that draws out the real moment, then the place and people come from their answer. It is their story.\n"
    "- Update the world from their latest message. If they change something, change only that.\n"
    "- Fill small gaps a photographer would need (age, clothes, light), but do not invent story or relationships. "
    "look is never empty, so every person can appear right away.\n"
    "- people are only the others in the scene, never the user themselves: the user goes in `me`, as a short description "
    "(their job or relation to the others). Only people the user asked for exist. At most 4. Give a real name if they gave none.\n"
    "- say: one or two short sentences. Acknowledge what just appeared, then ask for the single most useful missing thing: "
    "the place, then who is there, then what they want to happen. Never ask two things.\n"
    "- chips: tappable answers to your question, concrete to this world, in the user's voice.\n"
    "- ready: true once there is a place, at least one person and a goal. Say they can begin whenever they like.\n"
    "- stuck: true when they say they don't know, have no idea, or seem lost. Then reassure them in one line; templates will be shown.\n"
    "- If they say they want to begin but something is missing, tell them warmly what you still need, in one line. "
    "If they tell you to decide, fill it in yourself.\n"
    "Reply with a single JSON object and nothing else."
)


def build_turn(llm: LLM, world: World, history: list[tuple[str, str]], message: str, ui: str, name: str = "") -> BuildReply:
    talk = "\n".join(f"{who}: {text}" for who, text in history[-12:]) or "(just started)"
    return llm.ask(
        BuildReply, GUIDE,
        (f"The user's name is {name}.\n" if name else "")
        + f"World so far:\n{world.model_dump_json()}\n\nConversation:\n{talk}\n\nUser: {message}\n\n"
        f"Write say, chips, title, roles and captions in language '{ui}'. language is what the people in the scene speak.\n"
        'Return {"world": {"title": "", "place": "", "language": "", "me": "", "goal": "", "people": '
        '[{"name": "", "role": "", "gender": "m|f", "look": "", "manner": ""}]}, "say": "", "chips": [], "ready": false, "stuck": false}',
    )


SELF = {"나", "저", "본인", "me", "i", "myself", "user", "自分", "私", "僕"}


def others(w: World) -> World:
    """The user stands in the scene as themselves, not as a character on the stage."""
    return w.model_copy(update={"people": [p for p in w.people if p.name.strip().lower() not in SELF][:4]})


def as_situation(w: World) -> str:
    people = "\n".join(f"- {p.name}: {p.role}. Look: {p.look}. Manner: {p.manner}" for p in w.people)
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
