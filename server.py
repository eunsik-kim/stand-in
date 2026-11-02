"""Web stage: camera + microphone in the browser, the scene on the server.

Usage: uv run server.py   then open http://localhost:8010 in Chrome
(speech recognition needs Chrome; camera analysis runs locally in the page).
"""

import base64
import json
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from standin import admin, auth, voice
from standin.ending import ending
from standin.art import background, background_prompt, portraits, scene, scene_edit, scene_from
from standin.builder import Person, World, as_situation, build_turn, lock, others
from standin.coach import coach_turn
from standin.compiler import compile_scenario, structural_problems
from standin.debrief import debrief
from standin.engine import Line, Scene, open_scene, simulated_user, stream_step, FOREIGN
from standin.llm import LLM
from standin.signals import Signals, describe
from standin.spec import SPOKEN, Scenario
from standin.store import DailySpend, Store

load_dotenv()
ROOT = Path(__file__).parent
PRESETS = ROOT / "presets"
SAVE = ROOT / "runs" / "web"
SAVE.mkdir(parents=True, exist_ok=True)
ART = ROOT / "runs" / "art"  # images made for custom scenes
ART.mkdir(parents=True, exist_ok=True)
WORLDS = ROOT / "runs" / "worlds"  # where worlds were kept before the store; moved in once at start
WORLDS.mkdir(parents=True, exist_ok=True)
TTS = ROOT / "runs" / "tts"
TTS.mkdir(parents=True, exist_ok=True)

WRITER_MODEL = "muse-spark-1.3"
ACTOR_MODEL = "muse-spark-1.3"
CHECK_MODEL = "muse-spark-1.3"
IMAGE_MODEL = "muse-image-1.0"

store = Store(ROOT / "runs" / "standin.db")  # accounts, today's usage and spend, worlds: what must outlive a restart
moved = store.import_world_files(WORLDS)
if moved:
    print(f"moved {moved} saved worlds into the store", flush=True)
spend = DailySpend(store, float(os.getenv("STANDIN_SERVER_BUDGET", "3.0")), who=admin.current.get)  # dollars per UTC day
cache = ROOT / "runs" / "cache"
env = os.getenv
writer = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.7, "writer")
actor = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.8, "actor", "minimal")
checker = LLM(env("STANDIN_CHECK_MODEL", CHECK_MODEL), spend, cache, 0.0, "checker", "minimal")
quick_writer = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.7, "writer", "low")
coach = LLM(env("STANDIN_WRITER_MODEL", WRITER_MODEL), spend, cache, 0.2, "coach")
interviewer = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.5, "builder", "low")
painter = LLM(env("STANDIN_IMAGE_MODEL", IMAGE_MODEL), spend, cache, 1.0, "painter")
whisper = LLM(env("STANDIN_ACTOR_MODEL", ACTOR_MODEL), spend, cache, 0.3, "live_coach", "minimal")


class Session:
    def __init__(self, scenario: Scenario, preset: str | None, reveal: str, ui: str, owner: str):
        self.id = secrets.token_urlsafe(12)  # 96 random bits; the owner check below is what actually guards a scene
        self.touched = time.time()
        self.scene: Scene = open_scene(scenario)
        self.preset = preset
        self.reveal = reveal  # how much of the cast the user chose to see beforehand
        self.ui = ui
        self.goals = ["todo"] * len(scenario.success_criteria)
        self.assets: dict = {}
        self.owner = owner
        self.history: list[tuple] = []  # state before each turn, for "say that again"
        self.voiceable: set[tuple[str, str]] = set()  # (speaker, sentence) as streamed: the only text /api/speak will voice
        self.turn = 0
        self.lock = threading.Lock()


sessions: dict[str, Session] = {}
SESSIONS_PER_OWNER = 5  # opening a sixth scene lets the oldest one go


def own_session(sid: str, request: Request) -> Session:
    """The caller's own scene, or 404: knowing a scene's id is not enough to act on someone else's."""
    s = sessions.get(sid)
    if not s or (auth.on() and s.owner != visitor(request)):
        raise HTTPException(404, "unknown session")
    s.touched = time.time()
    return s


def versioned(url: str) -> str:
    """Tie an image URL to its file, so a remade portrait is never hidden by an old browser copy."""
    return f"{url}?v={int((PRESETS / 'img' / Path(url).name).stat().st_mtime)}"


CATEGORY = {"mystery": {"detective", "suspect", "holmes", "lupin", "celeste", "tip"}, "history": {"eiffel", "tesla", "seondal"},
            "sim": {"lawyer", "judge", "auction", "anchor", "pitch", "deal", "debate", "exit", "apartment"},
            "family": {"toddler", "santa", "son", "husband", "daughter", "worry", "dinner", "groupchat", "zerowon", "bet", "wordtrap", "table"},
            "love": {"trip", "date", "confess", "inlaws", "breakup", "reunion", "fishing", "manofhonor", "bestie", "ring", "overheard", "list"}}


def category(preset_id: str) -> str:
    name = preset_id.split("_", 1)[-1]
    return next((c for c, names in CATEGORY.items() if name in names), "sim")


def load_presets() -> dict[str, dict]:
    out = {p.stem: json.loads(p.read_text()) for p in sorted(PRESETS.glob("*.json"))}
    for preset in out.values():
        assets = preset.get("assets", {})
        if "background" in assets:
            assets["background"] = versioned(assets["background"])
        for name, faces in assets.get("portraits", {}).items():
            assets["portraits"][name] = {expr: versioned(url) for expr, url in faces.items()}
        assets["shots"] = {turn: versioned(url) for turn, url in assets.get("shots", {}).items()}
        for clue in assets.get("clues", {}).values():
            clue["img"] = versioned(clue["img"])
    return out


def public_line(ln: Line) -> dict:
    """What the stage may show during the scene: no thoughts, no mood numbers."""
    return {"turn": ln.turn, "speaker": ln.speaker, "say": ln.say, "expression": ln.expression}


def public_scenario(s: Scenario) -> dict:
    return {"title": s.title, "language": s.language, "setting": s.setting, "user_role": s.user_role,
            "user_goal": s.user_goal, "user_knows": s.user_knows, "max_turns": s.max_turns, "success_criteria": s.success_criteria,
            "cast": [{"name": c.name, "role": c.role, "appearance": c.appearance, "voice": c.voice, "enters_at": c.enters_at}
                     for c in s.cast]}


def draft(scenario: Scenario, assets: dict, preset: str | None, gate1=None) -> dict:
    """The full scene for the briefing/editor. The page hides private fields unless the user opens them."""
    return {"preset": preset, "scenario": scenario.model_dump(), "assets": assets, "gate1": gate1 or [],
            "background_prompt": background_prompt(scenario)}


if os.getenv("STANDIN_REQUIRE_LOGIN") and (not auth.on() or len(os.getenv("STANDIN_SESSION_SECRET", "")) < 32):
    # on the public site a missing key must stop the server, not quietly open it to anyone
    raise SystemExit("STANDIN_REQUIRE_LOGIN is set, but GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET or a 32+ character STANDIN_SESSION_SECRET is missing")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(auth.router)  # Google sign-in; does nothing until its keys are set
app.include_router(admin.router(store, spend, lambda: {"scenes": len(sessions), "worlds_open": len(builds)}))
gate = admin.Gate(store)  # each account's daily allowance, on top of the per-address limit below


# per visitor address, per hour: enough for real use, not enough to drain the credits
LIMITS = {"/api/speak": 300, "/api/image": 20}
PAID_PER_HOUR = int(os.getenv("STANDIN_HOURLY_LIMIT", "150"))
recent: dict[tuple[str, str], deque] = defaultdict(deque)


def over_limit(request: Request) -> bool:
    path = request.url.path
    if request.method != "POST" or not path.startswith("/api/") or path in ("/api/start", "/api/template", "/api/build/finish"):  # free: nothing is generated
        return False
    who = request.headers.get("fly-client-ip") or (request.client.host if request.client else "")
    limit = LIMITS.get(path, PAID_PER_HOUR)
    hits, now = recent[who, path if path in LIMITS else "llm"], time.time()
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= limit:
        return True
    hits.append(now)
    return False


def visitor(request: Request) -> str:
    """Whose worlds and runs these are: the Google account (`auth.user(request)`) when sign-in is on,
    else a random id kept in the browser."""
    if auth.on():
        account = auth.user(request)
        return account["sub"] if account else "!signed-out"  # matches no owner
    return request.headers.get("x-visitor", "")[:40]


BODY_LIMIT = {"/api/episode": 4_000_000, "/api/start": 256_000}  # a photo; a whole scenario; everything else is small
swept = 0.0


def sweep() -> None:
    """Scenes and worlds nobody has touched for six hours leave memory (worlds stay in the store)."""
    global swept
    now = time.time()
    if now - swept < 300:
        return
    swept = now
    for sid in [k for k, s in sessions.items() if now - s.touched > 6 * 3600]:
        sessions.pop(sid, None)
    for bid in [k for k, b in builds.items() if now - b.touched > 6 * 3600]:
        builds.pop(bid, None)


@app.middleware("http")
async def guard(request, call_next):
    if request.method == "POST" and request.url.path.startswith(("/api/", "/auth/")):
        size = request.headers.get("content-length")
        if size is None or not size.isdigit() or int(size) > BODY_LIMIT.get(request.url.path, 64_000):
            return JSONResponse({"detail": "That request is too large."}, status_code=413)
    sweep()
    if auth.on() and request.method == "POST" and request.url.path.startswith("/api/") and not auth.user(request):
        return JSONResponse({"detail": "Sign in with Google first."}, status_code=401)
    if over_limit(request):
        return JSONResponse({"detail": "Too many requests for now. Please try again in a while."}, status_code=429)
    stop = gate.check(request)  # also marks whose request this is, so its spend lands on that account
    if stop:
        return stop
    response = await call_next(request)
    gate.settle(request, response.status_code)
    spend.flush()  # voices add to the total without a model call; write them now
    # Portraits keep their file names when remade, so the browser must check for a new version.
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
             "cast": [c["name"] for c in v["scenario"]["cast"]], "assets": v.get("assets", {}), "category": category(k)}
            for k, v in load_presets().items()]


# ---------- building a world by talking ----------
class Build:
    def __init__(self):
        self.id = uuid.uuid4().hex[:10]
        self.world = World()
        self.talk: list[tuple[str, str]] = []
        self.ui = "en"
        self.owner = ""
        self.ready = False
        self.shots: dict[str, str] = {}  # turn -> photograph, prepared ahead for a template
        self.background = ""
        self.extras: dict = {}  # a template's clues and what it's based on
        self.extras_place = ""
        self.shots_for = ""  # the world as it was when the photos and clues were made for it
        self.faces: dict[str, dict] = {}  # name -> that person's reactions in the same photograph
        self.drawn: dict[str, str] = {}  # what each image was drawn from, to redraw only on a real change
        self.prepared: dict[str, Future] = {}  # world as JSON -> its scene, written before the user asks
        self.touched = time.time()

    def save(self) -> None:
        """Into the store, with the hidden side once it is written, so a reopened story keeps its secrets, photos and clues."""
        key = self.world.model_dump_json()
        done = self.prepared.get(key)
        scenario = done.result().model_dump() if done and done.done() and not done.exception() else None
        store.save_world(self.id, self.owner, {
            "id": self.id, "owner": self.owner, "ready": self.ready, "ui": self.ui, "world": self.world.model_dump(), "talk": self.talk,
            "background": self.background, "faces": self.faces, "drawn": self.drawn, "shots": self.shots, "extras": self.extras,
            "extras_place": self.extras_place, "shots_for": self.shots_for, "scenario": scenario, "scenario_for": key if scenario else ""})


builds: dict[str, Build] = {}
easel = ThreadPoolExecutor(6)


def find_build(bid: str, owner: str) -> Build | None:
    """Worlds live in the store, so leaving or restarting never loses one. Only their maker can open them."""
    if bid in builds:
        b = builds[bid]
        if b.owner != owner:
            return None
        b.touched = time.time()
        return b
    data = store.load_world(bid, owner) if bid else None
    if not data:
        return None
    b = Build()
    b.owner = owner
    b.id, b.world, b.background, b.faces, b.drawn = (data["id"], World.model_validate(data["world"]), data["background"],
                                                     data["faces"], data["drawn"])
    b.talk = [tuple(x) for x in data["talk"]]
    b.ready = data.get("ready", False)
    b.ui = data.get("ui", b.ui)
    b.shots, b.extras = data.get("shots", {}), data.get("extras", {})
    b.extras_place, b.shots_for = data.get("extras_place", ""), data.get("shots_for", "")
    if data.get("scenario"):  # the hidden side it was written with, so secrets and turning points stay the same
        done: Future = Future()
        done.set_result(Scenario.model_validate(data["scenario"]))
        b.prepared[data["scenario_for"]] = done
    builds[b.id] = b
    draw_changes(b, b.world)  # worlds saved before they had a photograph get one now
    return b


def prepare(b: Build) -> Future:
    """Write the hidden side as soon as the world can start, so beginning is instant."""
    key = b.world.model_dump_json()
    if key not in b.prepared:
        world = b.world.model_copy(deep=True)
        b.prepared[key] = easel.submit(lambda: lock(compile_scenario(
            quick_writer, checker, as_situation(world), ui=b.ui, locked=True, repairs=1, log=lambda _: None)[0], world))
        b.prepared[key].add_done_callback(lambda _: b.save())  # keep the hidden side with the world
    return b.prepared[key]


def draw_changes(b: Build, w: World, redraw: bool = False) -> None:
    """One photograph holds the place and the people. It changes only when someone is added or removed or the
    user asks, and always by editing it, so nobody's face ever changes on its own."""
    if not w.place:
        return
    names = ",".join(sorted(p.name for p in w.people))
    has_scene = "_scene" in b.background
    if b.background and b.drawn.get("names") == names and not redraw and (has_scene or not w.people):
        return
    b.drawn["names"] = names
    version = b.drawn["version"] = str(int(b.drawn.get("version", 0)) + 1)

    def paint(place: str, people: list[Person], base: str) -> None:
        try:
            draw(place, people, base)
        except Exception as e:  # the conversation goes on; the next change tries again
            print(f"drawing failed for {b.id}: {e!r:.200}", flush=True)
            b.drawn.pop("names", None)

    def draw(place: str, people: list[Person], base: str) -> None:
        stem = uuid.uuid4().hex[:10]
        path = ART / f"{stem}_scene.jpg"
        if people and base:
            source = PRESETS / "img" / Path(base.split("?")[0]).name
            scene_edit(painter, source if source.exists() else ART / Path(base).name, place, people, path)
        elif people:
            scene(painter, place, people, path)
        else:
            path = ART / f"{stem}_bg.jpg"
            background(painter, ("Candid photograph from the viewer's own eyes, standing in this place: "
                                 f"{place} No people, no text, no logos. Natural light."), path)
        if b.drawn.get("version") == version:  # a newer change wins
            b.background = f"/art/{path.name}"
            b.save()

    easel.submit(paint, w.place, list(w.people), b.background if has_scene else "")


def build_state(b: Build) -> dict:
    return {"id": b.id, "world": b.world.model_dump(), "background": b.background, "faces": b.faces, "talk": b.talk, "ready": b.ready, "shots": b.shots}


START_OVER = re.compile(r"^\s*(다시 만들|처음부터|새로 만들|다시 할래|다시 시작|start over|restart|やり直)", re.I)


class BuildReq(BaseModel):
    id: str = Field("", max_length=64)
    message: str = Field(max_length=2000)
    ui: str = Field("en", max_length=8)
    name: str = Field("", max_length=80)
    kickoff: bool = False


@app.post("/api/build")
def build(req: BuildReq, request: Request):
    b = find_build(req.id, visitor(request)) or Build()
    b.owner = visitor(request)
    builds[b.id] = b
    b.ui = req.ui
    message = req.message.strip()
    if req.kickoff:  # the user chose a template: the director speaks first
        message = ("(The user just picked this scene. Do not repeat the setting. In one warm line, ask the first thing that makes it "
                   "theirs: what is between them and the main person, or how they feel walking in.)")
    reply = build_turn(interviewer, b.world, b.talk, message, req.ui, req.name)
    if not req.kickoff and (reply.reset or START_OVER.search(message)):  # they said to start over, so this world is gone
        builds.pop(b.id, None)
        store.delete_world(b.id, b.owner)
        return {"reset": True, "say": reply.say}
    reply.world = others(reply.world)
    b.talk += ([] if req.kickoff else [("User", message)]) + [("You", reply.say)]
    spoken = sum(who == "User" for who, _ in b.talk)
    b.ready = b.ready or (reply.ready and spoken >= 3)  # a few real exchanges before anyone walks in
    b.world = reply.world
    draw_changes(b, b.world, reply.redraw)
    b.save()
    if b.world.place and b.world.people:
        prepare(b)
    print(f"build {b.id}: {len(b.world.people)} people | {spend.summary()}", flush=True)
    return {**build_state(b), "say": reply.say, "chips": reply.chips, "ready": b.ready, "stuck": reply.stuck, "begin": reply.begin}


@app.get("/api/history")
def history(request: Request):
    """This visitor's worlds to pick up again and past rehearsals, newest first."""
    me, worlds, runs = visitor(request), [], []
    for d in store.worlds_of(me, 12):
        if d["talk"]:
            worlds.append({"id": d["id"], "title": d["world"]["title"] or d["talk"][0][1][:40], "when": d["_updated"]})
    for path in sorted(SAVE.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
        d = json.loads(path.read_text())
        if d.get("owner", "") != me:
            continue
        crit = d["debrief"]["criteria"]
        runs.append({"id": path.stem, "title": d["scenario"]["title"], "when": path.stat().st_mtime,
                     "met": sum(c["met"] for c in crit), "all": len(crit)})
    return {"worlds": worlds, "runs": runs}


@app.get("/api/history/{rid}")
def past_run(rid: str, request: Request):
    path = SAVE / f"{Path(rid).name}.json"
    run = json.loads(path.read_text()) if path.exists() else None
    if not run or run.get("owner", "") != visitor(request):
        raise HTTPException(404, "unknown run")
    return run


@app.get("/api/build/{bid}")
def build_art(bid: str, request: Request):
    b = find_build(bid, visitor(request))
    if not b:
        raise HTTPException(404, "unknown build")
    return build_state(b)


class FinishReq(BaseModel):
    id: str = Field(max_length=64)
    ui: str = Field("en", max_length=8)


@app.post("/api/build/finish")
def build_finish(req: FinishReq, request: Request):
    """Usually already written by `prepare`; expressions keep arriving while the scene runs."""
    b = find_build(req.id, visitor(request))
    if not b:
        raise HTTPException(404, "unknown build")
    b.ui = req.ui
    try:
        if b.extras and b.world.place == b.extras_place:  # talking on a template's page must not rewrite the story
            scenario, rewritten = next(iter(b.prepared.values())).result(timeout=180), False
        else:  # a story rewritten around the user's changes: the old turning points' photos and clues would show the wrong people
            scenario, rewritten = prepare(b).result(timeout=180), b.world.model_dump_json() != b.shots_for
    except TimeoutError:
        raise HTTPException(503, "The story is taking too long to write. Please try again.")
    return draft(scenario, {"background": b.background, "scene": {}, "shots": {} if rewritten else b.shots,
                            **({} if rewritten else b.extras)}, None)


def load_scene(b: Build, s: Scenario) -> None:
    """A finished scene as a world: who is there at the start, anyone else is a surprise."""
    b.world = World(title=s.title, place=s.setting, language=s.language, me=s.user_role, goal=s.user_goal,
                    people=[Person(name=c.name, role=c.role, gender=c.gender, look=c.appearance, manner=c.voice, history=c.history)
                            for c in s.cast if not c.enters_at])
    done: Future = Future()
    done.set_result(s)
    b.shots_for = b.world.model_dump_json()
    b.prepared[b.shots_for] = done


class EpisodeReq(BaseModel):
    idea: str = Field(max_length=2000)
    genre: str = Field("", max_length=80)
    image: str = Field("", max_length=4_000_000)  # a picture the user attached, as a data URL
    ui: str = Field("en", max_length=8)


@app.post("/api/episode")
def episode(req: EpisodeReq, request: Request):
    """One idea becomes a full episode, as good as a template: people arrive, things turn, photos made ahead."""
    if not req.idea.strip():
        raise HTTPException(400, "empty idea")
    idea = f"{req.genre}: {req.idea.strip()}" if req.genre else req.idea.strip()
    s = compile_scenario(quick_writer, checker, idea, ui=req.ui, episode=True, repairs=1, log=lambda _: None)[0]
    b = Build()
    b.owner, b.ui = visitor(request), req.ui
    load_scene(b, s)
    as_person = lambda c: Person(name=c.name, role=c.role, gender=c.gender, look=c.appearance)
    here = [c for c in s.cast if not c.enters_at]
    stem = uuid.uuid4().hex[:10]
    first = ART / f"{stem}_scene.jpg"
    people = [as_person(c) for c in here]
    try:
        if req.image:
            scene_from(painter, base64.b64decode(req.image.split(",", 1)[-1]), s.setting, people, first)
        else:
            scene(painter, s.setting, people, first)
    except Exception:  # the content filter misfires on ordinary scenes
        scene(painter, s.setting.split(".")[0] + ".", people, first)
    b.background = f"/art/{first.name}"
    b.drawn["names"] = ",".join(sorted(person.name for person in b.world.people))

    def turning_points() -> None:  # keeps going while the user talks, so nobody waits for it
        current, present = first, list(here)
        for e in sorted(s.events, key=lambda x: x.at_turn):
            present += [c for c in s.cast if c.name in e.enters]
            path = ART / f"{stem}_{e.at_turn}_scene.jpg"
            try:
                scene_edit(painter, current, "the same room", [as_person(c) for c in present], path, e.shot)
                b.shots[str(e.at_turn)] = f"/art/{path.name}"
                current = path
            except Exception as err:
                print(f"shot failed: {err!r:.160}", flush=True)
    easel.submit(turning_points)
    builds[b.id] = b
    b.save()
    return build_state(b)


class TemplateReq(BaseModel):
    id: str = Field(max_length=64)
    ui: str = Field("en", max_length=8)


@app.post("/api/template")
def template(req: TemplateReq, request: Request):
    """A template is a world someone already built: same screen, same conversation, ready to begin."""
    p = load_presets().get(req.id)
    if not p:
        raise HTTPException(404, "unknown template")
    s = Scenario.model_validate(p["scenario"])
    b = Build()
    b.owner, b.ui = visitor(request), req.ui
    load_scene(b, s)
    assets = p.get("assets", {})
    b.background, b.shots = assets.get("background", ""), assets.get("shots", {})
    b.extras = {k: assets[k] for k in ("clues", "basis") if k in assets}
    b.extras_place = b.world.place
    b.drawn["names"] = ",".join(sorted(person.name for person in b.world.people))
    builds[b.id] = b
    return build_state(b)


class StartReq(BaseModel):
    scenario: dict  # possibly edited by the user on the briefing screen
    assets: dict = {}
    preset: str | None = Field(None, max_length=64)
    reveal: str = Field("profile", max_length=16)  # names | profile | all
    ui: str = Field("en", max_length=8)


@app.post("/api/start")
def start(req: StartReq, request: Request):
    try:
        scenario = Scenario.model_validate(req.scenario)
    except ValueError as e:
        raise HTTPException(400, f"scenario is invalid: {e}")
    problems = [p for p in structural_problems(scenario) if "secret" not in p]  # the user may spoil their own scene
    if problems:
        raise HTTPException(400, "; ".join(problems))
    s = Session(scenario, req.preset, req.reveal, req.ui, visitor(request))
    s.assets = req.assets
    mine = sorted((x for x in sessions.values() if x.owner == s.owner), key=lambda x: x.touched)
    for old in mine[:max(0, len(mine) - SESSIONS_PER_OWNER + 1)]:
        sessions.pop(old.id, None)
    sessions[s.id] = s
    print(f"start {s.id}: {scenario.title} (reveal={req.reveal}) | {spend.summary()}", flush=True)
    return {"id": s.id, "scenario": public_scenario(scenario), "assets": req.assets,
            "lines": [public_line(s.scene.lines[0])]}


class TurnReq(BaseModel):
    id: str = Field(max_length=64)
    say: str = Field(max_length=2000)
    signals: Signals = Signals()
    auto: bool = False
    evidence: str = Field("", max_length=80)  # whose clue the player laid on the table


def take_back(s, index: int) -> None:
    size, moods, revealed, left, turn, goals, arrived, beat, advance = s.history[index]
    del s.history[index:]
    sc = s.scene
    del sc.lines[size:]
    sc.moods, sc.revealed, sc.left, sc.ended = moods, revealed, left, ""
    sc.arrived, sc.beat, sc.advance, sc.verdict = arrived, beat, advance, None
    s.turn, s.goals = turn, goals


@app.post("/api/turn")
def turn(req: TurnReq, request: Request):
    s = own_session(req.id, request)
    if not req.say.strip():
        raise HTTPException(400, "empty line")

    def events():
        with s.lock:
            if s.scene.ended:
                yield json.dumps({"type": "done", "ended": s.scene.ended, "cues": ""}) + "\n"
                return
            sc = s.scene
            s.history.append((len(sc.lines), {k: m.model_copy() for k, m in sc.moods.items()},
                              set(sc.revealed), set(sc.left), s.turn, list(s.goals), set(sc.arrived), sc.beat, sc.advance))
            s.turn += 1
            cues = describe(req.signals, s.ui)
            t0, first = time.time(), True
            clue = s.assets.get("clues", {}).get(req.evidence)
            shown = f"{clue['name']} (it proves: {sc.scenario.character(req.evidence).secret})" if clue else ""
            try:
                for ev in stream_step(s.scene, s.turn, req.say.strip(), actor, cues=cues,
                                      signals=req.signals.model_dump(), auto=req.auto, shown=shown):  # raw numbers kept for the debrief
                    if ev["type"] == "sentence":
                        s.voiceable.add((ev["speaker"], ev["text"].strip()))  # before the page can ask for its voice
                    if first and ev["type"] == "sentence":
                        spend.seconds.setdefault("first_sentence", []).append(time.time() - t0)
                        first = False
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
            except Exception as e:  # the turn never happened: they can send it again
                print(f"turn failed {s.id}: {e!r}", flush=True)
                take_back(s, len(s.history) - 1)
                yield json.dumps({"type": "failed"}) + "\n"
                return
        print(f"turn {s.id}/{s.turn}: cues=[{cues}] | {spend.summary()} | {spend.latency()}", flush=True)

    return StreamingResponse(events(), media_type="application/x-ndjson")


class CoachReq(BaseModel):
    id: str = Field(max_length=64)
    say: str = Field(max_length=2000)


@app.post("/api/coach")
def live_coach(req: CoachReq, request: Request):
    s = own_session(req.id, request)
    c = coach_turn(whisper, s.scene, req.say, s.goals, s.ui)
    for i, g in enumerate(c.goals[:len(s.goals)]):  # progress only moves forward
        if s.goals[i] != "done" and not (s.goals[i] == "partial" and g.status == "todo"):
            s.goals[i] = g.status
    return {"goals": s.goals, "notes": [g.note for g in c.goals], "good": c.good, "hint": c.hint,
            "better": c.better, "why": c.why}


class RewindReq(BaseModel):
    id: str = Field(max_length=64)
    turn: int | None = None  # go back to just before this user turn; default is the latest one


@app.post("/api/rewind")
def rewind(req: RewindReq, request: Request):
    """Take back the last line, or return to any earlier moment and play it differently."""
    s = own_session(req.id, request)
    index = len(s.history) - 1 if req.turn is None else (req.turn or 0) - 1
    if not 0 <= index < len(s.history):
        raise HTTPException(400, "nothing to take back")
    with s.lock:
        take_back(s, index)
    sc = s.scene
    beats = sorted(sc.scenario.events, key=lambda e: e.at_turn)
    said = next((ln for ln in reversed(sc.lines) if ln.speaker not in ("user", "narration")), None)
    return {"turn": s.turn, "goals": s.goals, "found": len(sc.revealed), "shot": beats[sc.beat - 1].at_turn if sc.beat else 0,
            "last": {"speaker": said.speaker, "say": said.say, "expression": said.expression} if said else None}


class SpeakReq(BaseModel):
    id: str = Field(max_length=64)
    speaker: str = Field(max_length=80)
    text: str = Field(max_length=600)


@app.post("/api/speak")
def speak(req: SpeakReq, request: Request):
    s = own_session(req.id, request)
    scenario = s.scene.scenario
    c = next((c for c in scenario.cast if c.name == req.speaker), None)
    text = req.text.strip()
    if not c or not text:
        raise HTTPException(400, "unknown speaker or empty text")
    # only what this person said in this scene gets a voice, so the endpoint can't be used to voice anything else
    said = (c.name, text) in s.voiceable or any(ln.speaker == c.name and text in ln.say for ln in s.scene.lines)
    if not said:
        raise HTTPException(400, "only lines this person said can be voiced")
    if not voice.available():
        raise HTTPException(503, "no speech key")  # the page falls back to the browser voice
    return Response(voice.synthesize(text, voice.voice_for(scenario, c), spend, TTS), media_type="audio/mpeg")


class AutoReq(BaseModel):
    id: str = Field(max_length=64)
    tone: str = Field("", max_length=400)  # how this person likes to talk, set in onboarding


@app.post("/api/auto")
def autopilot(req: AutoReq, request: Request):
    """Autopilot: a stand-in speaks for the user until they take over."""
    s = own_session(req.id, request)
    mine = [ln.say for ln in s.scene.lines if ln.speaker == "user"]
    persona = (f"Speak in very short lines of a few words, the way this person really would, working toward: {s.scene.scenario.user_goal}. "
               f"How they talk: {req.tone or 'casual'}. Fit the time and place of the setting. "
               + (f"Their lines so far: {' / '.join(mine[-3:])}. Keep the same stance and level of politeness, and start differently from all of these. " if mine else "")
               + "Never quit the scene, and never reveal or solve what this person has not worked out yet: ask, push or stall instead.")
    return {"say": simulated_user(actor, persona)(s.scene, s.turn + 1).strip()}


class Replies(BaseModel):
    thought: str = ""
    replies: list[str]


@app.post("/api/replies")
def replies(req: AutoReq, request: Request):
    """Three things the user could say next, for when finding the words is the hard part."""
    s = own_session(req.id, request)
    sc = s.scene.scenario
    got = actor.ask(
        Replies, "You suggest what a person could say next in a live conversation. Reply with a single JSON object.",
        f"Setting: {sc.setting}\nThe person: {sc.user_role}\nWhat they want: {sc.user_goal}\nWhat winning takes: {'; '.join(sc.success_criteria)}\n"
        f"Turn {s.turn + 1} of {sc.max_turns}.\n\nTranscript:\n{s.scene.public()}\n\n"
        f"Write 3 different things they could say next, in language '{sc.language}', in the first person. {SPOKEN} "
        "Keep each one light and short, what someone actually blurts out: 2 to 10 words, naming people or things from the transcript exactly. "
        "The first one moves toward what winning takes: a thread left open, a new question or an offer. Never re-ask what the player just asked; "
        "if it was dodged, try another way in. "
        + ("Time is nearly up: if winning needs a decision, such as a ruling, a price or a yes or no, the first one makes it. " if s.turn + 3 >= sc.max_turns else "")
        +
        f"How this person talks, follow it closely: {req.tone or 'casual'}. "
        "Put the one you recommend first. Make them really different: push back, ask something, or dodge. "
        "No quotation marks. Do not repeat anything already said.\n"
        f"thought: what is going through this person's head right now, in the first person, one short natural sentence in language '{sc.language}', "
        "about what they just noticed and what they want to find out next. It holds only what they could know: never an answer, a verdict or advice.\n"
        'Return {"thought": "...", "replies": ["...", "...", "..."]}',
    )
    plain = (lambda x: FOREIGN.sub("", x).strip().strip('"“”\'')) if sc.language in ("ko", "en") else str.strip
    return {"thought": plain(got.thought), "replies": [plain(r) for r in got.replies[:3]]}


class EndReq(BaseModel):
    id: str = Field(max_length=64)


@app.post("/api/ending")
def closing_page(req: EndReq, request: Request):
    s = own_session(req.id, request)
    sc = s.scene
    e, verdict = ending(quick_writer, sc, s.ui)
    found = sc.revealed | ({c.name for c in sc.scenario.cast} & set(verdict.came_out))
    own = [ln for ln in sc.lines if ln.speaker == "user" and not ln.say.startswith("(")]  # a silence or a clue isn't a line to retake
    pivot = next((ln for ln in own if ln.turn == e.pivot_turn), None) or (own[-3] if len(own) >= 3 else None)
    won = all(verdict.met)
    best = "" if e.best_line.startswith("(") else e.best_line
    return {"title": e.title, "story": e.story, "best_line": best, "found": sorted(found),
            "outcome": "won" if won else "partial" if any(verdict.met) else "lost",
            "pivot": {"turn": pivot.turn, "line": pivot.say} if pivot and not won else None,
            "missed": [x.line for x in verdict.truths if x.name not in found],
            "goals": [{"text": c, "met": i < len(verdict.met) and verdict.met[i]} for i, c in enumerate(sc.scenario.success_criteria)]}


@app.post("/api/end")
def end(req: EndReq, request: Request):
    s = own_session(req.id, request)
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
        {**result, "owner": s.owner, "scenario": sc.scenario.model_dump(), "assets": s.assets},
        indent=2, ensure_ascii=False))
    print(f"end {s.id} | {spend.summary()}", flush=True)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8010")), log_level="warning")
