"""Prepare a ready-to-play scene ahead of time: scenario + background + portraits.

Usage:
  uv run make_preset.py landlord "집주인이 보증금에서 수리비 80만원을 빼겠다고 함. 못 받아들인다고 말해야 함"
  uv run make_preset.py salary            # existing preset: only (re)make missing images

Portraits: a neutral photo per character, with the other expressions edited from
it. Add --portraits to remake them for an existing preset.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import BadRequestError

from standin.art import background, background_prompt, portraits, scene, scene_edit
from standin.compiler import compile_scenario
from standin.llm import LLM, Spend
from standin.builder import Person
from standin.spec import Scenario

ROOT = Path(__file__).parent
PRESETS = ROOT / "presets"
IMG = PRESETS / "img"
WRITER_MODEL = "muse-spark-1.3"
CHECK_MODEL = "muse-spark-1.3"
IMAGE_MODEL = "muse-image-1.0"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("situation", nargs="?")
    ap.add_argument("--ui", default="en", help="language the user reads")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--episode", action="store_true", help="people arrive and twists land, with a photograph for each")
    ap.add_argument("--portraits", action="store_true", help="remake all portraits")
    ap.add_argument("--budget", type=float, default=0.60)
    args = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    spend = Spend(args.budget)
    cache = ROOT / "runs" / "cache"
    path = PRESETS / f"{args.id}.json"
    if path.exists():
        preset = json.loads(path.read_text())
    else:
        if not args.situation:
            raise SystemExit(f"no preset '{args.id}' yet; give a situation")
        writer = LLM(os.getenv("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.7, "writer")
        checker = LLM(os.getenv("STANDIN_CHECK_MODEL", CHECK_MODEL), spend, cache, 0.0, "checker", "low")
        scenario, gate1 = compile_scenario(writer, checker, args.situation, ui=args.ui, episode=args.episode)
        preset = {"situation": args.situation, "ui": args.ui, "scenario": scenario.model_dump(), "gate1": gate1}
        path.write_text(json.dumps(preset, indent=2, ensure_ascii=False))
        print(f"scenario: {scenario.title} ({len(scenario.cast)} people) | {spend.summary()}")

    s = Scenario.model_validate(preset["scenario"])
    if args.episode:
        painter = LLM(os.getenv("STANDIN_IMAGE_MODEL", IMAGE_MODEL), spend, cache, 1.0, "painter")
        here = [c for c in s.cast if not c.enters_at]
        as_person = lambda c: Person(name=c.name, role=c.role, gender=c.gender, look=c.appearance)
        current = IMG / f"{args.id}_scene.jpg"
        for attempt in range(3):  # the content filter misfires on ordinary scenes, so ask again
            try:
                place = s.setting if attempt == 0 else s.setting.split(".")[0] + "."
                scene(painter, place, [as_person(c) for c in here], current)
                break
            except BadRequestError:
                if attempt == 2:
                    raise
        shots = {}
        for e in s.events:
            here += [c for c in s.cast if c.name in e.enters]
            if e.shot or e.enters:
                next_shot = IMG / f"{args.id}_{e.at_turn}_scene.jpg"
                try:
                    scene_edit(painter, current, s.setting, [as_person(c) for c in here], next_shot, e.shot)
                except BadRequestError:
                    continue  # the scene carries on with the last photograph
                shots[str(e.at_turn)] = f"/presets/img/{next_shot.name}"
                current = next_shot
        preset["assets"] = {"background": f"/presets/img/{args.id}_scene.jpg", "shots": shots}
        path.write_text(json.dumps(preset, indent=2, ensure_ascii=False))
        print(f"done {path} | {len(shots)} shots | {spend.summary()}")
        return
    if not args.no_images:
        painter = LLM(os.getenv("STANDIN_IMAGE_MODEL", IMAGE_MODEL), spend, cache, 1.0, "painter")
        assets = preset.setdefault("assets", {})
        if "background" not in assets:
            background(painter, background_prompt(s), IMG / f"{args.id}_bg.jpg")
            assets["background"] = f"/presets/img/{args.id}_bg.jpg"
            print(f"background | {spend.summary()}")
        faces = {} if args.portraits else assets.setdefault("portraits", {})
        assets["portraits"] = faces
        for i, c in enumerate(s.cast):
            if c.name not in faces:
                faces[c.name] = portraits(painter, s.setting, c.gender, c.appearance, c.role, IMG, f"{args.id}_{i}", "/presets/img")
                print(f"portraits: {c.name} | {spend.summary()}")
                path.write_text(json.dumps(preset, indent=2, ensure_ascii=False))  # keep progress
        path.write_text(json.dumps(preset, indent=2, ensure_ascii=False))
    print(f"done {path} | {spend.summary()}")


if __name__ == "__main__":
    main()
