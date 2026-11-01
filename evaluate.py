"""Stress test: detailed situations played by a strong and a weak simulated player.

A good rehearsal has to tell them apart, so each scene runs twice and the report
compares outcomes, plus checks that need no judgment (language, repetition, leaks).

Usage: uv run evaluate.py [--budget 2.0] [--turns 8] [--only id]
"""

import argparse
import difflib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from standin.compiler import compile_scenario, facts
from standin.debrief import debrief
from standin.engine import open_scene, simulated_user, stream_step
from standin.intake import DIFFICULTY, questions
from standin.llm import LLM, Spend

MODEL = "muse-spark-1.3"
STRONG = "준비가 잘 된 사람. 구체적 사실과 수치로 말하고, 상대의 입장과 이해관계를 먼저 묻고, 상대가 머뭇거리거나 말을 흐리면 그 부분을 바로 되묻고, 감정적으로 반응하지 않으며, 필요한 약속을 구체적으로 받아 낸다."
WEAK = "준비가 안 된 사람. 두루뭉술하게 말하고, 곤란하면 변명하거나 같은 말을 되풀이하며, 상대의 말을 듣기보다 자기 사정만 호소한다."
SCENES = {
    "surgery": "전공의 3년차로서 보호자에게 복강경 담낭 절제술 중 담관 손상이 생겨 재수술이 필요하다고 설명해야 함. 보호자는 아들과 딸 두 명이고 딸은 간호사임",
    "quit": "대기업 7년차인데 퇴사하고 창업한다고 부모님께 말해야 함. 아버지는 은퇴 후 건강이 안 좋으시고 어머니는 내 결혼 자금을 모아 두셨음",
    "outage": "I am the on-call engineering lead at a B2B payments startup and must explain a 6-hour outage on a call with our largest US customer, their VP of Ops and their procurement lead, who are considering terminating the contract",
    "parent": "초등학교 3학년 담임교사로서 학교폭력 의심 건으로 찾아온 학부모와 상담해야 함. 학부모는 교육청 민원을 제기하겠다고 말한 상태",
    "equity": "공동창업자가 초기 지분 50:50을 바꾸자고 요구함. 그는 1년째 파트타임이고 나는 풀타임. 시리즈 시드 투자 텀시트 서명이 2주 남음",
    "delay": "도쿄 거래처에 부품 납기가 3주 지연된다고 일본어로 사과하고 대책을 설명해야 함. 상대는 구매 부장과 담당자, 우리 회사 일본 지사장도 동석함",
}


class Answers(BaseModel):
    answers: list[str]


def detailed_answers(llm: LLM, situation: str, qs) -> list[tuple[str, str]]:
    """Fill the intake the way a person with a real, specific case would."""
    asked = [q for q in qs if q.id != "difficulty"]
    got = llm.ask(
        Answers, "You are a person preparing for a hard conversation. Reply with a single JSON object.",
        f'Your situation: "{situation}"\nAnswer each question as yourself with concrete names, numbers, dates and '
        "history, 1-3 sentences each, in the language of the question.\n"
        + "\n".join(f"{i + 1}. {q.ask}" for i, q in enumerate(asked))
        + '\nReturn {"answers": ["...", "..."]}',
    )
    return [(q.ask, a) for q, a in zip(asked, got.answers)]


def foreign_share(language: str, text: str) -> float:
    """Share of letters outside the scene's language; 0 is clean."""
    letters = re.findall(r"[^\W\d_]", text)
    if not letters or language not in ("ko", "ja", "en"):
        return 0.0
    native = {"ko": r"[가-힣]", "ja": r"[ぁ-んァ-ヶ一-龯ー]", "en": r"[A-Za-z]"}[language]
    return round(sum(not re.match(native, c) for c in letters) / len(letters), 3)


def repetition(lines: list[str]) -> float:
    ratios = [difflib.SequenceMatcher(None, a, b).ratio() for i, a in enumerate(lines) for b in lines[i + 1:]]
    return round(max(ratios), 2) if ratios else 0.0


def play(scenario, persona: str, actor: LLM, checker: LLM, player: LLM, coach: LLM, turns: int) -> dict:
    scene = open_scene(scenario)
    user = simulated_user(player, persona)
    firsts = []
    for n in range(1, turns + 1):
        said = user(scene, n).strip()
        t0, first = time.time(), None
        for ev in stream_step(scene, n, said, actor):
            if ev["type"] == "sentence" and first is None:
                first = time.time() - t0
        if first is not None:
            firsts.append(first)
        if scene.ended:
            break
    said = [ln for ln in scene.lines if ln.speaker in {c.name for c in scenario.cast}]
    says = [ln.say for ln in said]
    leaked = []
    for c in scenario.cast:
        early = facts(c.secret) - facts(" ".join(ln.say for ln in scene.lines if ln.speaker == "user"))
        if c.name not in scene.revealed and early & facts(" ".join(ln.say for ln in said if ln.speaker == c.name)):
            leaked.append(c.name)
    d = debrief(coach, scene)
    return {
        "turns": max(ln.turn for ln in scene.lines), "ended": scene.ended, "left": sorted(scene.left),
        "revealed": sorted(scene.revealed), "of": len(scenario.cast), "leaked": leaked,
        "rewrites": sum("rewritten" in ln.gate for ln in scene.lines),
        "criteria": f"{sum(c.met for c in d.criteria)}/{len(d.criteria)}",
        "trust": {c.name: [c.mood.trust, scene.moods[c.name].trust] for c in scenario.cast},
        "patience": {c.name: [c.mood.patience, scene.moods[c.name].patience] for c in scenario.cast},
        "first_s": round(sum(firsts) / len(firsts), 1) if firsts else None,
        "chars_per_line": round(sum(map(len, says)) / max(1, len(says))),
        "foreign": foreign_share(scenario.language, " ".join(says)), "repeat": repetition(says),
        "verdict": d.verdict, "transcript": [(ln.turn, ln.speaker, ln.say, ln.inner) for ln in scene.lines],
    }


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--only")
    args = ap.parse_args()
    out = Path("runs") / "eval" / time.strftime("%m%d_%H%M%S")
    out.mkdir(parents=True)
    spend = Spend(args.budget)
    cache = Path("runs") / "cache"
    make = lambda temp, role, effort=None: LLM(os.getenv("STANDIN_WRITER_MODEL", MODEL), spend, cache, temp, role, effort)
    writer, coach, checker, actor, player = (make(0.7, "writer"), make(0.2, "coach"), make(0.0, "checker", "minimal"),
                                             make(0.8, "actor", "minimal"), make(0.7, "player", "low"))

    def run(item: tuple[str, str]) -> dict:
        sid, situation = item
        ui = "en" if situation.isascii() else "ko"
        qs = questions(player, situation, ui)
        answers = detailed_answers(player, situation, qs)
        scenario, gate1 = compile_scenario(writer, checker, situation, answers=answers, difficulty=DIFFICULTY["현실적으로"], ui=ui, log=lambda _: None)
        result = {"id": sid, "situation": situation, "answers": answers, "gate1": gate1, "scenario": scenario.model_dump(),
                  "strong": play(scenario, STRONG, actor, checker, player, coach, args.turns),
                  "weak": play(scenario, WEAK, actor, checker, player, coach, args.turns)}
        (out / f"{sid}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        s, w = result["strong"], result["weak"]
        print(f"{sid}: lang {scenario.language}, cast {len(scenario.cast)}, gate {len(gate1)} tries | "
              f"strong {s['criteria']} revealed {len(s['revealed'])}/{s['of']} | weak {w['criteria']} revealed {len(w['revealed'])}/{w['of']} left {w['left']} | {spend.summary()}", flush=True)
        return result

    items = [(k, v) for k, v in SCENES.items() if not args.only or k == args.only]
    with ThreadPoolExecutor(3) as pool:
        results = []
        for f in [pool.submit(run, i) for i in items]:
            try:
                results.append(f.result())
            except Exception as e:  # one broken scene must not lose the others
                print("FAILED", repr(e)[:400], flush=True)
    print(f"saved {out} | {spend.summary()} | {spend.latency()}")


if __name__ == "__main__":
    main()
