"""Record the demo footage from the live site: screen frames, every voice clip and the moment it played.

Three parts, each its own run:
  dinner   "Pot Roast and Pink Slips": a blunt first take that goes wrong, its ending,
           "Go back to that moment", and a second take that lands
  own      a scene made from one sentence
  stories  a mystery, a piece of history, a first date, and the language picker

Onboarding happens before the capture starts. Frames come from the DevTools
screencast; voice clips are caught as the page plays them, since a screen capture
has no sound. Subtitles, clues, goal toasts and element boxes are logged so the
edit can find moments and zoom on them.

Usage: uv run --no-project --with playwright python demo/record.py dinner|own|stories [--url URL] [--out DIR]
Then:  uv run --no-project python demo/cut.py
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = Path(os.environ["LOCALAPPDATA"]) / "ms-playwright/chromium-1134/chrome-win/chrome.exe"

BLUNT = [  # take 1: how it goes when you just blurt it out
    "I quit my job. I'm starting a company. It's decided, so please don't start.",
    "Dad, you never believed in anything I did. Why would this be different?",
    "Josh, this isn't a joke. Stay out of it.",
    "Mom, I don't need your money or your permission.",
    "I'm not calling my old boss. Forget it.",
    "You always do this. Every single time.",
    "Fine. Think whatever you want.",
    "I'm done talking about it.",
    "Can we just eat?",
    "Whatever.",
]
BETTER = [  # take 2, from the moment the ending points back to
    "Dad, can I ask you something first? Did you ever want to quit and start something of your own?",
    "Mom, what worries you most? What would actually help you feel better about this?",
    "Josh, what's going on with your own job right now?",
    "I quit, and I'm starting the company. I won't take that back, but I'd like to do it with your eyes on it.",
    "Mom, if I show you a plan with six months of savings, would you go through it with me this week?",
    "Dad, would you tell me how it went when you tried? I want to learn from it.",
    "Thank you. All of you. This went better than I was afraid it would.",
    "Let's eat before it gets cold.",
    "Same time next Sunday?",
]

HOOKS = """
(() => {
  try { localStorage.setItem("standin-speed", "1.25"); localStorage.setItem("standin-pick", "0"); } catch (e) {}
  if (window.speechSynthesis) {  // no browser voices headless: end narration after a reading pause
    speechSynthesis.speak = u => setTimeout(() => u.onend && u.onend(), Math.max(2200, (u.text || "").length * 62) / (u.rate || 1));
    speechSynthesis.cancel = () => {};
  }
  const send = (kind, data) => window.__rec && window.__rec(kind, JSON.stringify(data));
  const now = () => Date.now() / 1000;
  const make = URL.createObjectURL;
  URL.createObjectURL = function (obj) {
    const url = make.call(URL, obj);
    if (obj instanceof Blob && obj.type.startsWith("audio")) {
      obj.arrayBuffer().then(buf => {
        const b = new Uint8Array(buf); let s = "";
        for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
        send("audio", {url, b64: btoa(s)});
      });
    }
    return url;
  };
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    const el = this;
    if (el.src.startsWith("blob:") && !el.__rec) {
      el.__rec = true;
      el.addEventListener("playing", () => send("play", {url: el.src, t: now(), rate: el.playbackRate}), {once: true});
      const stop = () => send("stop", {url: el.src, t: now(), pos: el.currentTime});
      el.addEventListener("ended", stop, {once: true});
      el.addEventListener("pause", stop, {once: true});
    }
    return play.apply(this, arguments);
  };
  // what appears on stage, with times: subtitles (who says what), clues, goal toasts
  const watch = () => {
    const cap = document.getElementById("caption"), clues = document.getElementById("clues"), toast = document.getElementById("goalToast");
    if (!cap || !clues || !toast) return setTimeout(watch, 300);
    let last = "";
    new MutationObserver(() => {
      const text = cap.innerText.trim();
      if (text && text !== last) { last = text; send("caption", {t: now(), text}); }
    }).observe(cap, {childList: true, subtree: true, characterData: true});
    new MutationObserver(ms => ms.forEach(m => m.addedNodes.forEach(n => n.dataset && send("clue", {t: now(), who: n.dataset.who}))))
      .observe(clues, {childList: true});
    new MutationObserver(() => toast.classList.contains("show") && send("toast", {t: now(), text: toast.textContent}))
      .observe(toast, {attributes: true, attributeFilter: ["class"]});
  };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", watch) : watch();
})();
"""

LISTENING = "() => { const c = document.querySelector('#choices .reply'); const t = document.getElementById('typed'); return !!c && !t.disabled; }"
ENDED = "() => { const e = document.getElementById('endCard'); return e && !e.classList.contains('hidden') && !e.classList.contains('waiting'); }"
LISTEN_OR_END = f"() => ({LISTENING})() || ({ENDED})()"


class Rec:
    """One browser run: the page, its screencast, and the log the edit reads."""

    def __init__(self, pw, out: Path):
        self.out = out
        (out / "frames").mkdir(parents=True, exist_ok=True)
        (out / "clips").mkdir(parents=True, exist_ok=True)
        for old in [*(out / "frames").glob("*.jpg"), *(out / "clips").glob("*.mp3")]:
            old.unlink()
        self.events, self.frames, self.clips = [], [], {}
        self.browser = pw.chromium.launch(executable_path=str(CHROME), args=[
            "--autoplay-policy=no-user-gesture-required", "--force-device-scale-factor=1.5"])
        ctx = self.browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5, locale="en-US")
        ctx.expose_function("__rec", self.on_rec)
        ctx.add_init_script(HOOKS)
        self.page = ctx.new_page()
        self.cdp = ctx.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self.on_frame)

    def on_rec(self, kind, payload):
        data = json.loads(payload)
        if kind == "audio":
            name = f"{len(self.clips):03d}.mp3"
            (self.out / "clips" / name).write_bytes(base64.b64decode(data["b64"]))
            self.clips[data["url"]] = name
        else:
            self.events.append({"kind": kind, **data})
            if kind in ("clue", "toast"):
                print(f"   {kind}: {data.get('who') or data.get('text')}", flush=True)

    def on_frame(self, p):
        (self.out / "frames" / f"{len(self.frames):06d}.jpg").write_bytes(base64.b64decode(p["data"]))
        self.frames.append(p["metadata"]["timestamp"])
        self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})

    def start(self):
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 90, "maxWidth": 1920, "maxHeight": 1080, "everyNthFrame": 1})

    def mark(self, label, **extra):
        self.events.append({"kind": "mark", "label": label, "t": time.time(), **extra})
        print(f"[{time.strftime('%H:%M:%S')}] {label}", flush=True)

    def box(self, label, selector):
        """Where something sits on screen (CSS px of the 1280x720 page), for zooming on it later."""
        r = self.page.evaluate("s => { const e = document.querySelector(s); if (!e) return null;"
                               " const b = e.getBoundingClientRect(); return [b.x, b.y, b.width, b.height]; }", selector)
        if r and r[2] and r[3]:
            self.events.append({"kind": "box", "label": label, "t": time.time(), "rect": r})

    def wait(self, ms):
        self.page.wait_for_timeout(ms)

    def save(self):
        self.cdp.send("Page.stopScreencast")
        self.wait(400)
        (self.out / "timeline.json").write_text(json.dumps(
            {"frames": self.frames, "events": self.events, "clips": self.clips}, indent=1))
        plays = sum(e["kind"] == "play" for e in self.events)
        print(f"{len(self.frames)} frames, {len(self.clips)} clips, {plays} played -> {self.out}")
        self.browser.close()


def onboard(r: Rec, url: str):
    p = r.page
    p.goto(url)
    p.wait_for_selector("#onboard:not(.hidden)")
    p.click("button[data-lang=en]")
    p.check("#voiceOut")
    p.uncheck("#voiceIn")
    p.click("#o1next")
    p.fill("#oname", "Alex")
    p.locator("#otones button").nth(1).click()
    p.click("#o2next")
    for i in (0, 3, 6):
        p.locator("#otraits button").nth(i).click()
    p.click("#o3next")
    p.wait_for_selector("#presets .card", timeout=30000)
    p.wait_for_load_state("networkidle")
    r.wait(1500)


def enter_scene(r: Rec):
    """From the briefing into the scene, past the intro card and the how-it-works card, until it's our turn."""
    p = r.page
    r.box("enter", "#benter")
    p.click("#benter")
    r.mark("scene")
    deadline = time.time() + 120
    while time.time() < deadline and not p.evaluate(LISTENING):
        if p.is_visible("#intro"):
            r.box("intro", "#intro")
            r.wait(3000)
            if p.is_visible("#intro"):
                p.click("#intro")
        if p.is_visible("#tourOk"):
            r.wait(1500)
            p.click("#tourOk")
        r.wait(300)
    p.wait_for_function(LISTENING, timeout=60000)


def say(r: Rec, label, text):
    p = r.page
    if p.is_visible("#clueBig"):
        r.wait(1800)
    r.box(f"{label}_choices", "#choices")
    r.mark(f"{label}_listen")
    r.wait(1200)
    p.click("#typed")
    r.box(f"{label}_typed", ".dock")
    p.locator("#typed").press_sequentially(text, delay=26)
    r.wait(450)
    p.keyboard.press("Enter")
    r.mark(f"{label}_sent", text=text)


def play_until_end(r: Rec, take, lines):
    for i, text in enumerate(lines, 1):
        r.page.wait_for_function(LISTEN_OR_END, timeout=180000)
        if r.page.evaluate(ENDED):
            break
        say(r, f"{take}_{i}", text)
    r.page.wait_for_function(ENDED, timeout=300000)
    r.mark(f"{take}_ending")
    for sel in ("#endCard .tcard", "#eGoals", "#eMissed", "#eBack"):
        r.box(f"{take}_{sel.split()[-1].lstrip('#.')}", sel)


def scroll_card(r: Rec, ms=6000):
    r.page.evaluate("""ms => new Promise(res => { const c = document.querySelector('#endCard .tcard');
      if (!c || c.scrollHeight <= c.clientHeight) return res();
      const end = c.scrollHeight - c.clientHeight, t0 = performance.now();
      const step = n => { const k = Math.min(1, (n - t0) / ms); c.scrollTop = end * k * k * (3 - 2 * k); k < 1 ? requestAnimationFrame(step) : res(); };
      requestAnimationFrame(step); })""", ms)


def dinner(r: Rec):
    p = r.page
    r.start()
    r.mark("home")
    r.wait(2500)
    card = p.locator("#presets .card", has_text="Pot Roast and Pink Slips")
    card.scroll_into_view_if_needed()
    r.wait(800)
    r.box("card", "#presets .card[data-id='en_dinner']")
    card.click()
    r.mark("briefing")
    p.wait_for_selector("#benter:not([disabled])", timeout=60000)
    r.box("cast", "#bpeople")
    r.box("world", "#bworld")
    r.wait(4500)
    enter_scene(r)

    play_until_end(r, "t1", BLUNT)
    r.wait(5000)
    scroll_card(r)
    r.wait(2500)
    if not p.is_visible("#eBack"):
        r.mark("no_go_back")
        return
    r.box("t1_eBack", "#eBack")
    p.locator("#eBack").scroll_into_view_if_needed()
    r.wait(1500)
    p.click("#eBack")
    r.mark("go_back")
    p.wait_for_function("() => document.getElementById('endCard').classList.contains('hidden')", timeout=30000)
    play_until_end(r, "t2", BETTER)
    r.wait(5000)
    scroll_card(r)
    r.wait(3000)


def nav(r: Rec, selector):
    """Sidebar buttons slide off screen on narrow windows; the menu button brings them back."""
    off = r.page.evaluate("s => { const b = document.querySelector(s).getBoundingClientRect();"
                          " return b.right <= 0 || b.left >= innerWidth || !b.width; }", selector)
    if off:
        r.page.click("#bmenu")
        r.wait(600)
    r.page.click(selector)


def own(r: Rec):
    p = r.page
    r.start()
    r.mark("home")
    r.wait(1500)
    r.box("own", "#bown")
    p.click("#bown")
    r.mark("own")
    r.wait(1200)
    p.click("#nidea")
    r.box("idea", "#bcustom .ncard")
    p.locator("#nidea").press_sequentially(
        "My landlord wants to keep $800 of my deposit for damage I didn't cause. I have to push back.", delay=30)
    r.wait(700)
    p.click("#ngo")
    r.mark("gen_start")
    p.wait_for_selector("#benter:not([disabled])", timeout=240000)
    r.mark("gen_done")
    r.box("gen_world", "#bworld")
    r.wait(4000)
    enter_scene(r)
    r.mark("gen_scene")
    r.wait(6000)


def stories(r: Rec):
    p = r.page
    r.start()
    r.mark("home")
    r.wait(1200)
    for cat, title in (("cat_mystery", "The Red-Headed League"), ("cat_history", "Sold: One Eiffel Tower"), ("cat_love", "Blind Date")):
        chip = p.locator("#pcats button", has_text={"cat_mystery": "Mystery", "cat_history": "History", "cat_love": "Love"}[cat])
        chip.click()
        r.mark(f"{cat}")
        r.wait(1500)
        p.locator("#presets .card", has_text=title).click()
        r.mark(f"story_{title}")
        p.wait_for_selector("#benter:not([disabled])", timeout=60000)
        r.wait(2800)
        nav(r, "#btplBtn")
        p.wait_for_selector("#presets .card", timeout=30000)
        r.wait(800)
    nav(r, "#bsettings")
    r.mark("languages")
    p.wait_for_selector("#o0:not(.hidden)", timeout=10000)
    r.box("langs", ".olangs")
    r.wait(2500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["dinner", "own", "stories"])
    ap.add_argument("--url", default="https://stand-in.fly.dev/")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or Path("demo/out/raw") / args.part
    with sync_playwright() as pw:
        r = Rec(pw, out)
        try:
            onboard(r, args.url)
            {"dinner": dinner, "own": own, "stories": stories}[args.part](r)
        finally:
            r.mark("end")
            r.save()


if __name__ == "__main__":
    main()
