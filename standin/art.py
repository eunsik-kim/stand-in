"""Scene images: a background and four expressions per character.

One neutral portrait is generated, then each other expression is an edit of that
photo, which keeps the person identical. The prompt asks for an unretouched phone
snapshot because polished, centered studio faces are what read as generated.
"""

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import BadRequestError
from PIL import Image

from .llm import LLM, edit_image, generate_image
from .spec import Scenario

EXPRESSIONS = {
    "pleased": "a small, polite, natural smile",
    "skeptical": "openly doubtful: one eyebrow clearly raised, eyes narrowed, a slight one-sided smirk, head tilted back a little",
    "annoyed": "irritated: brows pulled hard together, eyes narrowed, mouth a tight flat line, not sad, not pouting",
}


def background_prompt(s: Scenario) -> str:
    return (f"Photorealistic wide shot of an empty place, seen from where a person would sit or stand: {s.setting} "
            "No people, no text, no logos. Natural light, slightly soft focus, cinematic, calm colors.")


def background(painter: LLM, prompt: str, path: Path) -> None:
    img = Image.open(io.BytesIO(generate_image(painter, prompt, aspect_ratio="16:9"))).convert("RGB")
    img.thumbnail((1600, 1600))
    img.save(path, quality=85)


def portrait_prompt(setting: str, gender: str, appearance: str, role: str) -> str:
    gender = {"m": "man", "f": "woman"}.get(gender, "person")
    return (
        f"Candid unretouched smartphone photo of an ordinary real {gender}, taken by someone sitting across from them. "
        "Shot from about three meters away, at a polite conversational distance, the person seated and framed from the "
        "waist up, centered. The person is the clear subject and fills most of the frame height: the head takes up "
        "about one quarter of the image height, with some room around them. Relaxed posture, looking at the camera. Natural uneven light, mild sensor noise, "
        f"blurry real surroundings of this place: {setting[:160]} "
        "Real skin with pores and small unevenness, a slightly asymmetrical face. Not a model, no beauty filter, no "
        "studio light, no retouching, no 3D render look, no text. "
        f"Show their exact age, do not make them look older than stated. The person: {appearance} Role: {role}."
    )


def scene_prompt(place: str, people: list) -> str:
    who = "; ".join(f"{p.name}: {'man' if p.gender == 'm' else 'woman' if p.gender == 'f' else 'person'}, {p.role}, {p.look}" for p in people)
    return (
        "Candid unretouched photograph taken from the viewer's own eyes, as if the viewer is right there in the conversation, "
        f"at about 1.5 meters. The place: {place} In the place, naturally part of it, sharing its exact light and shadows: {who}. "
        "They look straight at the viewer, sitting or standing the way a person really would there. The people are the subject and fill the "
        "middle of the frame, with the surroundings all around them. Real skin, natural imperfect faces, not models, mild sensor noise. "
        "No text, no borders, no logos."
    )


def scene(painter: LLM, place: str, people: list, path: Path) -> None:
    img = Image.open(io.BytesIO(generate_image(painter, scene_prompt(place, people), aspect_ratio="16:9"))).convert("RGB")
    img.thumbnail((1792, 1792))
    img.save(path, quality=88)


def scene_expressions(painter: LLM, source: Path, person, folder: Path, stem: str, url: str) -> dict[str, str]:
    """The same photograph with only this person's face changed, so a reaction never breaks the room."""
    image = source.read_bytes()

    def change(item: tuple[str, str]) -> tuple[str, str]:
        expr, look = item
        prompt = (f"Same photograph, same people, place, framing and light. Only change the facial expression of {person.name} "
                  f"({person.look}) to: {look}. Everything else stays exactly the same.")
        try:
            data = edit_image(painter, image, prompt)
        except BadRequestError:  # the content filter misfires now and then on ordinary photos
            data = edit_image(painter, image, prompt)
        Image.open(io.BytesIO(data)).convert("RGB").save(folder / f"{stem}_{expr}.jpg", quality=88)
        return expr, f"{url}/{stem}_{expr}.jpg"

    with ThreadPoolExecutor(3) as pool:
        return dict(pool.map(change, EXPRESSIONS.items()))


def _tile(data: bytes, path: Path) -> None:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    side = min(img.size)
    x, y = (img.width - side) // 2, (img.height - side) // 2
    img.crop((x, y, x + side, y + side)).resize((1024, 1024), Image.LANCZOS).save(path, quality=88)


def portraits(painter: LLM, setting: str, gender: str, appearance: str, role: str,
              folder: Path, stem: str, url: str, on_neutral=None) -> dict[str, str]:
    neutral = generate_image(painter, portrait_prompt(setting, gender, appearance, role), aspect_ratio="1:1")
    _tile(neutral, folder / f"{stem}_neutral.jpg")
    if on_neutral:  # the person can appear while their other expressions are still being made
        on_neutral(f"{url}/{stem}_neutral.jpg")

    def change(item: tuple[str, str]) -> None:
        expr, look = item
        prompt = ("Same photo of the same person: same face, age, hair, clothes, camera, lighting and background. "
                  f"Only change the facial expression to: {look}. Keep the skin texture and photo noise.")
        try:
            data = edit_image(painter, neutral, prompt)
        except BadRequestError:  # the content filter misfires now and then on ordinary portraits
            data = edit_image(painter, neutral, prompt)
        _tile(data, folder / f"{stem}_{expr}.jpg")

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(change, EXPRESSIONS.items()))
    return {expr: f"{url}/{stem}_{expr}.jpg" for expr in ["neutral", *EXPRESSIONS]}
