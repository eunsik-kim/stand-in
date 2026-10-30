"""Scene images: a background and one 2x2 expression sheet per character.

The sheet holds the same face four times (neutral, pleased, skeptical, annoyed)
and is cut into tiles, so expressions stay consistent and cost one call.
"""

import io
from pathlib import Path

from PIL import Image

from .llm import LLM, generate_image
from .spec import Character, Scenario

EXPRESSIONS = ["neutral", "pleased", "skeptical", "annoyed"]  # sheet order: TL, TR, BL, BR


def background_prompt(s: Scenario) -> str:
    return (f"Photorealistic wide shot of an empty place, seen from where a person would sit or stand: {s.setting} "
            "No people, no text, no logos. Natural light, slightly soft focus, cinematic, calm colors.")


def background(painter: LLM, prompt: str, path: Path) -> None:
    img = Image.open(io.BytesIO(generate_image(painter, prompt, aspect_ratio="16:9"))).convert("RGB")
    img.thumbnail((1600, 1600))
    img.save(path, quality=85)


def portraits(painter: LLM, s: Scenario, c: Character, folder: Path, stem: str, url: str) -> dict[str, str]:
    data = generate_image(painter, (
        "A 2x2 grid of four photorealistic head-and-shoulders portraits of the SAME person, identical face, hair, "
        "clothes and lighting in all four, plain softly blurred office-neutral background, looking at the camera. "
        "Top-left: neutral, attentive. Top-right: slight warm smile. Bottom-left: skeptical, one eyebrow raised. "
        "Bottom-right: annoyed, frowning. No text, no borders, no labels. "
        f"The person: {c.appearance}. Role: {c.role}. Setting context: {s.setting[:200]}"
    ), aspect_ratio="1:1")
    sheet = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = sheet.size
    out = {}
    for k, expr in enumerate(EXPRESSIONS):
        x, y = (k % 2) * w // 2, (k // 2) * h // 2
        m = int(0.02 * w)  # trim seams between tiles
        tile = sheet.crop((x + m, y + m, x + w // 2 - m, y + h // 2 - m)).resize((384, 384), Image.LANCZOS)
        name = f"{stem}_{expr}.jpg"
        tile.save(folder / name, quality=88)
        out[expr] = f"{url}/{name}"
    sheet.save(folder / f"{stem}_sheet.jpg", quality=80)  # kept for eyeballing the crop
    return out
