"""Full-screen cards for the demo video, drawn as HTML and saved as 1920x1080 PNGs so text stays sharp.

Usage: uv run --no-project --with playwright python demo/cards.py   -> demo/out/cards/*.png
"""
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = Path(os.environ["LOCALAPPDATA"]) / "ms-playwright/chromium-1134/chrome-win/chrome.exe"
IMG = Path("presets/img").resolve().as_uri()

STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=block" rel="stylesheet">
<style>
:root{--bg:#0f1115;--fg:#eef0f4;--mute:#8d94a3;--acc:#f2c14e;--line:#2a2f3a;--good:#4cc38a;--think:#7aa2ff}
*{box-sizing:border-box}
html,body{margin:0;width:1920px;height:1080px;overflow:hidden;background:var(--bg);color:var(--fg);
  font-family:Inter,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.center{position:absolute;inset:0;display:grid;place-content:center;text-align:center;gap:28px;
  background:radial-gradient(circle at 50% 42%,#262012 0%,var(--bg) 62%)}
.logo{font-size:168px;font-weight:800;letter-spacing:-6px;line-height:1}
.tag{font-size:48px;font-weight:600;color:var(--acc);letter-spacing:-.5px}
.sub{font-size:30px;color:var(--mute)}
.photo{position:absolute;inset:0;background-size:cover;background-position:center}
.photo::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,transparent 45%,rgba(8,9,12,.92))}
.label{position:absolute;left:110px;bottom:100px;z-index:1;font-size:76px;font-weight:700;letter-spacing:-1.5px;
  text-shadow:0 4px 30px rgba(0,0,0,.6)}
.label small{display:block;font-size:30px;font-weight:600;color:var(--acc);letter-spacing:2px;text-transform:uppercase;margin-bottom:14px}
.kicker{position:absolute;left:110px;top:84px;font-size:30px;font-weight:700;color:var(--acc);letter-spacing:3px;text-transform:uppercase}
.d{position:absolute;border:2px solid var(--line);border-radius:28px;background:#14161c;padding:30px 34px;transition:none}
.d h3{margin:0 0 10px;font-size:34px;font-weight:700;letter-spacing:-.5px;line-height:1.2}
.d p{margin:0;font-size:24px;line-height:1.45;color:#c9cdd6}
.d .tagl{font-size:22px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--acc);margin-bottom:8px}
.chips{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px}
.chips span{font-size:22px;padding:7px 18px;border-radius:999px;background:#1d2029;border:1px solid var(--line);color:var(--fg)}
.arrow{position:absolute;height:4px;background:var(--line)}
.arrow::after{content:"";position:absolute;right:-4px;top:-9px;border:11px solid transparent;border-left:16px solid var(--line)}
.strip{display:flex;gap:22px;align-items:center;margin-top:18px}
.strip img{width:300px;height:169px;object-fit:cover;border-radius:14px;border:2px solid var(--line)}
.strip b{font-size:26px;color:var(--acc);font-weight:700}
.dim{opacity:.28}
.lit{border-color:var(--acc);box-shadow:0 0 0 2px var(--acc),0 18px 60px rgba(242,193,78,.18)}
.links{display:grid;gap:18px;justify-items:center}
.url{font-size:64px;font-weight:700;color:var(--acc);letter-spacing:-1px}
</style>"""


def page(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'>{STYLE}</head><body>{body}</body></html>"


def title() -> str:
    return page("""<div class="center"><div class="logo">Stand-In</div>
      <div class="tag">Rehearse the conversation before it happens.</div></div>""")


def problem(img: str, kicker: str, label: str) -> str:
    return page(f"""<div class="photo" style="background-image:url('{IMG}/{img}')"></div>
      <div class="label"><small>{kicker}</small>{label}</div>""")


def built(stage: int) -> str:
    """0: everything, 1: Muse Spark, 2: code, 3: Muse Image."""
    cls = lambda part: "" if stage == 0 else ("lit" if part == stage else "dim")
    side = "dim" if stage in (2, 3) else ""  # the input and output boxes belong to the Muse Spark step
    return page(f"""
      <div class="kicker" style="top:64px">How it works</div>
      <div class="d {side}" style="left:110px;top:190px;width:270px"><h3>You</h3><p>Type or speak.<br>Camera cues optional.</p></div>
      <div class="arrow {side}" style="left:392px;top:268px;width:52px"></div>
      <div class="d {cls(1)}" style="left:460px;top:140px;width:1000px">
        <div class="tagl">Muse Spark</div><h3>One streamed call plays the director and every character</h3>
        <p>Who answers, what they say, with a gesture and an expression. Streamed, so the first words come in about 3 seconds.</p>
        <div class="chips"><span>Director</span><span>Diane</span><span>Frank</span><span>Josh</span></div></div>
      <div class="arrow {side}" style="left:1472px;top:268px;width:52px"></div>
      <div class="d {side}" style="left:1540px;top:190px;width:270px"><h3>Voices</h3><p>ElevenLabs, one voice per person.</p></div>
      <div class="d {cls(2)}" style="left:460px;top:470px;width:1000px">
        <div class="tagl">Code, not the model</div><h3>Owns the moods, nudges the secrets out</h3>
        <div class="chips"><span>Moods clamped</span><span>Openness rises with trust</span><span>Replies cached</span><span>Budget cap</span></div></div>
      <div class="d {cls(3)}" style="left:110px;top:716px;width:1700px;padding:24px 34px">
        <div class="tagl">Muse Image</div><h3 style="margin:0">Paints the scene, then edits that same photo as the story turns</h3>
        <div class="strip"><img src="{IMG}/en_dinner_scene.jpg"><b>edit</b><img src="{IMG}/en_dinner_2_scene.jpg"><b>edit</b>
          <img src="{IMG}/en_dinner_4_scene.jpg"><b>edit</b><img src="{IMG}/en_dinner_6_scene.jpg"></div></div>""")


def end() -> str:
    return page("""<div class="center"><div class="logo">Stand-In</div>
      <div class="links"><div class="url">stand-in.fly.dev</div><div class="sub">github.com/eunsik-kim/stand-in</div></div>
      <div class="sub" style="margin-top:20px">Built with Meta Muse Spark and Muse Image &middot; voices by ElevenLabs</div></div>""")


CARDS = {
    "title": title(),
    "problem_1": problem("en_dinner_scene.jpg", "One shot", "Telling your parents you quit"),
    "problem_2": problem("en_exit_scene.jpg", "One shot", "Resigning"),
    "problem_3": problem("en_pitch_scene.jpg", "One shot", "Pitching investors"),
    **{f"built_{i}": built(i) for i in range(4)},
    "end": end(),
}


def main():
    out = Path("demo/out/cards").resolve()
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=str(CHROME))
        tab = browser.new_page(viewport={"width": 1920, "height": 1080})
        for name, html in CARDS.items():
            src = out / f"{name}.html"
            src.write_text(html, encoding="utf-8")
            tab.goto(src.as_uri())
            tab.wait_for_load_state("networkidle")
            tab.evaluate("document.fonts.ready")
            tab.screenshot(path=str(out / f"{name}.png"))
            print(name)
        browser.close()


if __name__ == "__main__":
    main()
