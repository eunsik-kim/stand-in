"""Web stage: camera + microphone in the browser, the scene on the server.

Usage: uv run server.py   then open http://localhost:8010 in Chrome
(speech recognition needs Chrome; camera analysis runs locally in the page).
"""

import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from standin import voice
from standin.art import background, background_prompt, portraits
from standin.coach import coach_turn
from standin.compiler import compile_scenario, structural_problems
from standin.intake import DIFFICULTY, questions
from standin.debrief import debrief
from standin.engine import Line, Scene, open_scene, stream_step
from standin.llm import LLM, Spend
from standin.signals import Signals, describe
from standin.spec import Scenario

load_dotenv()
ROOT = Path(__file__).parent
PRESETS = ROOT / "presets"
SAVE = ROOT / "runs" / "web"
SAVE.mkdir(parents=True, exist_ok=True)
ART = ROOT / "runs" / "art"  # images made for custom scenes
ART.mkdir(parents=True, exist_ok=True)
TTS = ROOT / "runs" / "tts"
TTS.mkdir(parents=True, exist_ok=True)

WRITER_MODEL = "muse-spark-1.3"
ACTOR_MODEL = "muse-spark-1.3"
CHECK_MODEL = "muse-spark-1.3"
IMAGE_MODEL = "muse-image-1.0"

spend = Spend(float(os.getenv("STANDIN_SERVER_BUDGET", "3.0")))
cache = ROOT / "runs" / "cache"
env = os.getenv
writer = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.7, "writer")
actor = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.8, "actor", "minimal")
checker = LLM(env("STANDIN_CHECK_MODEL", CHECK_MODEL), spend, cache, 0.0, "checker", "minimal")
coach = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.2, "coach")
interviewer = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.4, "interviewer")
painter = LLM(env("STANDIN_IMAGE_MODEL", IMAGE_MODEL), spend, cache, 1.0, "painter")
whisper = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.3, "live_coach", "minimal")


class Session:
    def __init__(self, scenario: Scenario, preset: str | None, reveal: str, ui: str):
        self.id = uuid.uuid4().hex[:10]
        self.scene: Scene = open_scene(scenario)
        self.preset = preset
        self.reveal = reveal  # how much of the cast the user chose to see beforehand
        self.ui = ui
        self.goals = ["todo"] * len(scenario.success_criteria)
        self.history: list[tuple] = []  # state before each turn, for "say that again"
        self.turn = 0
        self.lock = threading.Lock()


sessions: dict[str, Session] = {}


def versioned(url: str) -> str:
    """Tie an image URL to its file, so a remade portrait is never hidden by an old browser copy."""
    return f"{url}?v={int((PRESETS / 'img' / Path(url).name).stat().st_mtime)}"


def load_presets() -> dict[str, dict]:
    out = {p.stem: json.loads(p.read_text()) for p in sorted(PRESETS.glob("*.json"))}
    for preset in out.values():
        assets = preset.get("assets", {})
        if "background" in assets:
            assets["background"] = versioned(assets["background"])
        for name, faces in assets.get("portraits", {}).items():
            assets["portraits"][name] = {expr: versioned(url) for expr, url in faces.items()}
    return out


def public_line(ln: Line) -> dict:
    """What the stage may show during the scene: no thoughts, no mood numbers."""
    return {"turn": ln.turn, "speaker": ln.speaker, "say": ln.say, "expression": ln.expression}


def public_scenario(s: Scenario) -> dict:
    return {"title": s.title, "language": s.language, "setting": s.setting, "user_role": s.user_role,
            "user_goal": s.user_goal, "max_turns": s.max_turns, "success_criteria": s.success_criteria,
            "cast": [{"name": c.name, "role": c.role, "appearance": c.appearance, "voice": c.voice}
                     for c in s.cast]}


def draft(scenario: Scenario, assets: dict, preset: str | None, gate1=None) -> dict:
    """The full scene for the briefing/editor. The page hides private fields unless the user opens them."""
    return {"preset": preset, "scenario": scenario.model_dump(), "assets": assets, "gate1": gate1 or [],
            "background_prompt": background_prompt(scenario)}


app = FastAPI()


@app.middleware("http")
async def always_revalidate(request, call_next):
    """Portraits keep their file names when remade, so the browser must check for a new version."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
(PRESETS / "img").mkdir(parents=True, exist_ok=True)
app.mount("/presets/img", StaticFiles(directory=PRESETS / "img"), name="preset_img")
app.mount("/art", StaticFiles(directory=ART), name="art")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/presets")
def presets():
    return [{"id": k, "ui": v.get("ui", "ko"), "title": v["scenario"]["title"], "situation": v["situation"],
             "cast": [c["name"] for c in v["scenario"]["cast"]], "assets": v.get("assets", {})}
            for k, v in load_presets().items()]


class IntakeReq(BaseModel):
    situation: str
    ui: str = "en"


@app.post("/api/intake")
def intake(req: IntakeReq):
    if not req.situation.strip():
        raise HTTPException(400, "empty situation")
    qs = questions(interviewer, req.situation.strip(), req.ui)
    print(f"intake: {len(qs)} questions | {spend.summary()}", flush=True)
    return {"questions": [q.model_dump() for q in qs]}


class Answer(BaseModel):
    id: str = ""
    ask: str
    answer: str = ""


class CompileReq(BaseModel):
    situation: str
    answers: list[Answer] = []
    material: str = ""  # pasted resume, abstract, lease...
    ui: str = "en"


@app.post("/api/compile")
def compile_(req: CompileReq):
    if not req.situation.strip():
        raise HTTPException(400, "empty situation")
    difficulty = next((DIFFICULTY.get(a.answer.strip(), "real") for a in req.answers if a.id == "difficulty"), "real")
    answers = [(a.ask, a.answer) for a in req.answers if a.id != "difficulty"]
    scenario, gate1 = compile_scenario(writer, checker, req.situation.strip(), req.material,
                                     answers=answers, difficulty=difficulty, ui=req.ui)
    print(f"compiled: {scenario.title} ({difficulty}) | {spend.summary()}", flush=True)
    return draft(scenario, {}, None, gate1)


@app.get("/api/preset/{pid}")
def preset(pid: str):
    p = load_presets().get(pid)
    if not p:
        raise HTTPException(404, "unknown preset")
    return draft(Scenario.model_validate(p["scenario"]), p.get("assets", {}), pid, p.get("gate1"))


class ImageReq(BaseModel):
    scenario: dict
    kind: str  # "background" | "portrait"
    name: str = ""  # character, for a portrait
    prompt: str = ""  # background only: the user's edited prompt


@app.post("/api/image")
def image(req: ImageReq):
    s = Scenario.model_validate(req.scenario)
    stem = uuid.uuid4().hex[:10]
    if req.kind == "background":
        background(painter, req.prompt.strip() or background_prompt(s), ART / f"{stem}_bg.jpg")
        out = {"background": f"/art/{stem}_bg.jpg"}
    elif req.kind == "portrait":
        c = next((c for c in s.cast if c.name == req.name), None)
        if not c:
            raise HTTPException(400, "unknown character")
        out = {"portrait": portraits(painter, s, c, ART, stem, "/art")}
    else:
        raise HTTPException(400, "kind must be background or portrait")
    print(f"image {req.kind} {req.name} | {spend.summary()}", flush=True)
    return out


class StartReq(BaseModel):
    scenario: dict  # possibly edited by the user on the briefing screen
    assets: dict = {}
    preset: str | None = None
    reveal: str = "profile"  # names | profile | all
    ui: str = "en"


@app.post("/api/start")
def start(req: StartReq):
    try:
        scenario = Scenario.model_validate(req.scenario)
    except ValueError as e:
        raise HTTPException(400, f"scenario is invalid: {e}")
    problems = [p for p in structural_problems(scenario) if "secret" not in p]  # the user may spoil their own scene
    if problems:
        raise HTTPException(400, "; ".join(problems))
    s = Session(scenario, req.preset, req.reveal, req.ui)
    sessions[s.id] = s
    print(f"start {s.id}: {scenario.title} (reveal={req.reveal}) | {spend.summary()}", flush=True)
    return {"id": s.id, "scenario": public_scenario(scenario), "assets": req.assets,
            "lines": [public_line(s.scene.lines[0])]}


class TurnReq(BaseModel):
    id: str
    say: str
    signals: Signals = Signals()


@app.post("/api/turn")
def turn(req: TurnReq):
    s = sessions.get(req.id)
    if not s:
        raise HTTPException(404, "unknown session")
    if not req.say.strip():
        raise HTTPException(400, "empty line")

    def events():
        with s.lock:
            if s.scene.ended:
                yield json.dumps({"type": "done", "ended": s.scene.ended, "cues": ""}) + "\n"
                return
            sc = s.scene
            s.history.append((len(sc.lines), {k: m.model_copy() for k, m in sc.moods.items()},
                              set(sc.revealed), set(sc.left), s.turn, list(s.goals)))
            s.turn += 1
            cues = describe(req.signals, s.ui)
            t0, first = time.time(), True
            for ev in stream_step(s.scene, s.turn, req.say.strip(), actor, checker, cues=cues,
                                  signals=req.signals.model_dump()):  # raw numbers kept for the debrief
                if first and ev["type"] == "sentence":
                    spend.seconds.setdefault("first_sentence", []).append(time.time() - t0)
                    first = False
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        print(f"turn {s.id}/{s.turn}: cues=[{cues}] | {spend.summary()} | {spend.latency()}", flush=True)

    return StreamingResponse(events(), media_type="application/x-ndjson")


class CoachReq(BaseModel):
    id: str
    say: str


@app.post("/api/coach")
def live_coach(req: CoachReq):
    s = sessions.get(req.id)
    if not s:
        raise HTTPException(404, "unknown session")
    c = coach_turn(whisper, s.scene, req.say, s.goals, s.ui)
    for i, g in enumerate(c.goals[:len(s.goals)]):  # progress only moves forward
        if s.goals[i] != "done" and not (s.goals[i] == "partial" and g.status == "todo"):
            s.goals[i] = g.status
    return {"goals": s.goals, "notes": [g.note for g in c.goals], "good": c.good, "hint": c.hint,
            "better": c.better, "why": c.why}


class RewindReq(BaseModel):
    id: str


@app.post("/api/rewind")
def rewind(req: RewindReq):
    """Take back the last line and answer again."""
    s = sessions.get(req.id)
    if not s or not s.history:
        raise HTTPException(400, "nothing to take back")
    with s.lock:
        size, moods, revealed, left, turn, goals = s.history.pop()
        sc = s.scene
        del sc.lines[size:]
        sc.moods, sc.revealed, sc.left, sc.ended = moods, revealed, left, ""
        s.turn, s.goals = turn, goals
    return {"turn": s.turn, "goals": s.goals, "found": len(sc.revealed)}


class SpeakReq(BaseModel):
    id: str
    speaker: str
    text: str


@app.post("/api/speak")
def speak(req: SpeakReq):
    s = sessions.get(req.id)
    if not s:
        raise HTTPException(404, "unknown session")
    scenario = s.scene.scenario
    c = next((c for c in scenario.cast if c.name == req.speaker), None)
    if not c or not req.text.strip():
        raise HTTPException(400, "unknown speaker or empty text")
    if not voice.available():
        raise HTTPException(503, "no speech key")  # the page falls back to the browser voice
    return Response(voice.synthesize(req.text.strip(), voice.voice_for(scenario, c), spend, TTS), media_type="audio/mpeg")


class EndReq(BaseModel):
    id: str


@app.post("/api/end")
def end(req: EndReq):
    s = sessions.get(req.id)
    if not s:
        raise HTTPException(404, "unknown session")
    with s.lock:
        sc = s.scene
        sc.ended = sc.ended or "user ended the scene"
        if len(sc.lines) < 2:
            raise HTTPException(400, "nothing to debrief yet")
        d = debrief(coach, sc, s.ui)
        lines = [asdict(ln) for ln in sc.lines]
        result = {
            "ended": sc.ended, "reveal": s.reveal, "goals": s.goals, "debrief": d.model_dump(), "lines": lines,
            "cast": [{"name": c.name, "role": c.role, "goal": c.goal, "secret": c.secret,
                      "reveal_when": c.reveal_when, "revealed": c.name in sc.revealed,
                      "start_mood": c.mood.model_dump(), "end_mood": sc.moods[c.name].model_dump()}
                     for c in sc.scenario.cast],
        }
    slug = re.sub(r"\W+", "_", sc.scenario.title)[:40]
    (SAVE / f"{s.id}_{slug}.json").write_text(json.dumps(
        {**result, "scenario": sc.scenario.model_dump(), "spend": spend.summary(), "latency": spend.latency()},
        indent=2, ensure_ascii=False))
    print(f"end {s.id} | {spend.summary()}", flush=True)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8010")), log_level="warning")
