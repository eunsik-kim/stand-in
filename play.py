"""Rehearse any situation in the terminal.

Usage:
  uv run play.py "다음 주 팀장한테 연봉 15% 올려 달라고 해야 함"
  uv run play.py "PhD defense on my paper" --material paper.txt
  uv run play.py "..." --auto "근거 없이 감정에 호소하는 사람"   # simulated player, for testing
Type /end to stop the scene.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from standin.compiler import compile_scenario
from standin.debrief import debrief
from standin.engine import Line, Scene, run_scene, simulated_user
from standin.llm import LLM, Spend, preflight

WRITER_MODEL = "muse-spark-1.3"  # compiler + debrief: few calls, quality matters
ACTOR_MODEL = "muse-spark-1.3"  # the live scene: every turn, latency matters
CHECK_MODEL = "muse-spark-1.3"  # gates
USER_MODEL = "muse-spark-1.3"  # simulated player for --auto


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("situation")
    ap.add_argument("--material", type=Path, help="text file the scene should draw on (resume, paper...)")
    ap.add_argument("--auto", metavar="PERSONA", help="let a simulated player take the user's turns")
    ap.add_argument("--turns", type=int, help="override max user turns")
    ap.add_argument("--budget", type=float, default=0.50)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or Path("runs") / time.strftime("%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs") / "cache"  # shared, so a rerun of the same situation replays for free
    spend = Spend(args.budget)
    env = os.getenv
    writer = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.7, "writer")
    actor = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.8, "actor", "minimal")
    checker = LLM(env("STANDIN_CHECK_MODEL", CHECK_MODEL), spend, cache, 0.0, "checker", "minimal")
    coach = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.2, "coach")
    player = LLM(env("STANDIN_USER_MODEL", USER_MODEL), spend, cache, 0.7, "player") if args.auto else None

    for m in {c.model: c for c in (writer, actor, checker, player) if c}.values():
        preflight(m)
    print("preflight ok:", spend.summary())

    material = args.material.read_text() if args.material else ""
    print("compiling scenario...")
    scenario, gate1 = compile_scenario(writer, checker, args.situation, material)
    if args.turns:
        scenario.max_turns = args.turns
    (out / "scenario.json").write_text(scenario.model_dump_json(indent=2))
    print(f"\n== {scenario.title} ==\n{scenario.setting}\nYou are: {scenario.user_role}\n"
          f"Goal: {scenario.user_goal}\nIn the room:\n{scenario.public_cast()}\n")

    def show(ln: Line) -> None:
        if ln.speaker == "narration":
            print(f"   * {ln.say}")
        else:
            mood = f"  [{ln.expression} · trust {ln.mood['trust']} · patience {ln.mood['patience']}]"
            print(f"{ln.speaker}: {ln.say}{mood}")
            if args.auto and ln.inner:  # in a real session thoughts stay hidden until the debrief
                print(f"   (속마음: {ln.inner})" + (f"  <{ln.gate}>" if ln.gate else ""))

    if args.auto:
        auto = simulated_user(player, args.auto)

        def user_turn(scene: Scene, n: int) -> str:
            said = auto(scene, n)
            print(f"\nYOU> {said}")
            return said
    else:
        def user_turn(scene: Scene, n: int) -> str:
            return input(f"\n[{n}/{scenario.max_turns}] YOU> ")

    scene = run_scene(scenario, actor, user_turn, show)
    print(f"\n-- scene ended: {scene.ended} --\nwriting debrief...")
    d = debrief(coach, scene)

    print(f"\n== DEBRIEF ==\n{d.verdict}\n")
    for c in d.criteria:
        print(f"[{'O' if c.met else 'X'}] {c.criterion} (turn {c.turn}) - {c.why}")
    print("\n-- what they were really thinking --")
    for c in scenario.cast:
        print(f"{c.name}: goal: {c.goal}\n   secret: {c.secret}"
              f" ({'revealed' if c.name in scene.revealed else 'never revealed'})")
    for t in d.turning_points:
        print(f"\nturn {t.turn} · {t.character}\n  you: {t.what_user_said}\n  they thought: {t.what_they_thought}"
              f"\n  try: {t.try_instead}")

    record = {"situation": args.situation, "auto": args.auto, "gate1": gate1, "ended": scene.ended,
              "revealed": sorted(scene.revealed), "lines": [asdict(ln) for ln in scene.lines],
              "debrief": d.model_dump(), "spend": spend.summary(), "latency": spend.latency(),
              "models": {"writer": writer.model, "actor": actor.model, "checker": checker.model,
                         "player": player.model if player else None}}
    (out / "session.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\nspend: {spend.summary()}\nlatency: {json.dumps(spend.latency(), ensure_ascii=False)}\nsaved {out}")


if __name__ == "__main__":
    main()
