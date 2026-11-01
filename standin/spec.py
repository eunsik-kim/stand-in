"""Scenario spec: everything the stage, the cast and the director need.

Public fields are what the user may see before the scene. `goal`, `secret`
and `reveal_when` are private to that character (and the director) until the
debrief.
"""

from typing import Literal

from pydantic import BaseModel, Field

Expression = Literal["neutral", "pleased", "skeptical", "annoyed"]

# every line anyone says aloud follows this
SPOKEN = (
    "Spoken lines sound like real speech overheard, never written text. Short, unfinished, reacting in the moment: people trail off, "
    "restart, interrupt, use fillers and slang that fit who they are. No tidy full sentences, no explaining, no formal connectors, "
    "no summarising what was just said. In Korean use natural 구어체: contractions like 뭐, 걍, 근데, 아니, 어?, 그니까, endings like -거든, -잖아, -는데, -지 뭐, "
    "and 반말 or 존댓말 as the relationship really would; never 문어체 such as '-습니다만', '따라서', '그러나', '말씀드립니다' unless the person is truly that formal. "
    "Vary how people talk: never lean on the same filler or tag twice in a row, like 어? or 알지? or yeah?, and give each person their own rhythm. "
    "In English use contractions and fragments. In Japanese use casual spoken forms that fit the relationship."
)


class Mood(BaseModel):
    trust: int = Field(ge=0, le=100, description="how much they believe/like the user")
    patience: int = Field(ge=0, le=100, description="how long they will keep engaging")


class Character(BaseModel):
    name: str
    role: str = Field(description="public role in the scene, e.g. 'direct manager'")
    appearance: str = Field(description="for the portrait: age range, look, clothing")
    voice: str = Field(description="how they sound and talk: pace, register, verbal habits")
    gender: str = Field(default="", description="'m' or 'f', picks their speaking voice")
    history: str = Field(default="", description="what has already happened between them and the user; both remember it")
    enters_at: int = Field(default=0, description="user turn when they walk in; 0 means they are there from the start")
    goal: str = Field(description="PRIVATE: what they want out of this conversation")
    secret: str = Field(description="PRIVATE: a concrete fact the user does not know that would change the user's strategy")
    reveal_when: str = Field(description="PRIVATE: what the user must do/say before they let the secret slip")
    red_lines: list[str] = Field(description="things they will not do or accept")
    mood: Mood


class Event(BaseModel):
    at_turn: int = Field(ge=1, description="user turn after which the event happens")
    description: str = Field(description="what happens, written as stage narration")
    enters: list[str] = Field(default=[], description="names of people who walk in now")
    shot: str = Field(default="", description="how the photograph changes at this moment")


class Scenario(BaseModel):
    title: str
    language: str = Field(description="language the scene is played in, e.g. 'ko', 'en', 'ja'")
    setting: str = Field(description="place, time, atmosphere, time pressure")
    user_role: str
    user_goal: str
    user_knows: list[str] = Field(default=[], description="facts the user walks in knowing; never part of a secret")
    cast: list[Character]
    opening_speaker: str = Field(description="name of the character who speaks first")
    opening_line: str
    phases: list[str]
    events: list[Event]
    success_criteria: list[str] = Field(description="2-4 outcomes judgeable from the dialogue alone")
    max_turns: int = Field(ge=4, le=30, description="user turns before the scene is cut")

    def public_cast(self) -> str:
        return "\n".join(f"- {c.name}: {c.role}" for c in self.cast)

    def character(self, name: str) -> Character:
        return next(c for c in self.cast if c.name == name)
