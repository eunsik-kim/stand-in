"""Assemble the demo video from the recorded parts, the cards and the narration (see demo/STORYBOARD.md).

Recorded parts come from demo/record.py (demo/out/raw/<part>), cards from demo/cards.py,
narration from demo/narrate.py. Pictures are drawn frame by frame here (zoom, cross-fades,
badges) and piped to ffmpeg; the voices the page played and the narration are mixed after.

Usage: uv run --no-project --with pillow python demo/cut.py [--from 0 --to 999] [--out demo/out/stand-in-demo.mp4]
"""
import argparse
import bisect
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 30
XF = 0.4  # cross-fade between shots
OUT = Path("demo/out")
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 30)


def smooth(k):
    k = min(1.0, max(0.0, k))
    return k * k * (3 - 2 * k)


class Take:
    """A recorded part: frames with their times, the voice clips, and what happened when."""

    def __init__(self, name):
        self.dir = OUT / "raw" / name
        tl = json.loads((self.dir / "timeline.json").read_text())
        self.frames, self.events, self.clips = tl["frames"], tl["events"], tl["clips"]
        self.scale = Image.open(self.dir / "frames" / "000000.jpg").width / 1280
        self.marks = {}
        for e in self.events:
            if e["kind"] == "mark":
                self.marks.setdefault(e["label"], e["t"])
        caps = [e for e in self.events if e["kind"] == "caption"]
        self.lines, run = [], None  # subtitles arrive letter by letter: keep (first letter, full line, text)
        for c in caps:
            if run and c["text"].startswith(run[2]):
                run = (run[0], c["t"], c["text"])
            else:
                if run:
                    self.lines.append(run)
                run = (c["t"], c["t"], c["text"])
        if run:
            self.lines.append(run)
        stops = {}
        for e in self.events:
            if e["kind"] == "stop":
                stops.setdefault(e["url"], e)
        self.voices = []  # (start, end, clip file, rate, seconds of the clip that played)
        for e in self.events:
            if e["kind"] == "play" and e["url"] in self.clips:
                s = stops.get(e["url"])
                pos = s["pos"] if s else 8.0
                self.voices.append((e["t"], e["t"] + pos / e["rate"], self.dir / "clips" / self.clips[e["url"]], e["rate"], pos))

    def mark(self, label, after=0.0):
        return next(e["t"] for e in self.events if e["kind"] == "mark" and e["label"] == label and e["t"] >= after)

    def line(self, needle, after=0.0):
        """(when the line starts appearing, when it is complete) for the first subtitle containing needle."""
        return next((a, b) for a, b, text in self.lines if needle in text and a >= after)

    def first(self, kind, after=0.0):
        return next(e["t"] for e in self.events if e["kind"] == kind and e["t"] >= after)

    def box(self, label):
        """Centre of a logged element, in 1280x720 page pixels."""
        x, y, w, h = next(e["rect"] for e in self.events if e["kind"] == "box" and e["label"] == label)
        return x + w / 2, y + h / 2

    def frame(self, t):
        i = max(0, bisect.bisect_right(self.frames, t) - 1)
        return self.dir / "frames" / f"{i:06d}.jpg"

    def speaking(self, a, b):
        return any(s < b and e > a for s, e, *_ in self.voices)


@dataclass
class Clip:
    take: Take
    a: float
    b: float
    speed: float = 1.0
    zoom: list = field(default_factory=list)  # [(seconds into the shot, x, y, zoom)] in page pixels
    badge: str = ""
    voices: bool = True
    hold: float = 1.6  # a still screen longer than this, with nobody speaking, is shortened to it

    def __post_init__(self):
        fr, keep = self.take.frames, []
        lo = bisect.bisect_left(fr, self.a)
        cur = self.a
        for i in range(lo, len(fr)):
            g0, g1 = fr[i], fr[i + 1] if i + 1 < len(fr) else self.b
            g0, g1 = max(g0, self.a), min(g1, self.b)
            if g1 - g0 > self.hold and not self.take.speaking(g0, g1):
                keep.append((cur, g0 + self.hold))
                cur = g1
            if fr[i] >= self.b:
                break
        keep.append((cur, self.b))
        self.pieces = [(p, q) for p, q in keep if q > p]
        self.dur = sum(q - p for p, q in self.pieces) / self.speed

    def src(self, lt):
        s = lt * self.speed
        for p, q in self.pieces:
            if s <= q - p:
                return p + s
            s -= q - p
        return self.pieces[-1][1]

    def local(self, t):
        """Shot time for a take time, or None if that moment was cut."""
        acc = 0.0
        for p, q in self.pieces:
            if p <= t <= q:
                return (acc + t - p) / self.speed
            acc += q - p
        return None

    def view(self, lt):
        if not self.zoom:
            return 640.0, 360.0, 1.0
        keys = self.zoom
        if lt <= keys[0][0]:
            return keys[0][1:]
        for (t0, *v0), (t1, *v1) in zip(keys, keys[1:]):
            if lt <= t1:
                k = smooth((lt - t0) / max(t1 - t0, 1e-6))
                return tuple(a + (b - a) * k for a, b in zip(v0, v1))
        return keys[-1][1:]

    def image(self, lt):
        img = load(self.take.frame(self.src(lt)))
        x, y, z = self.view(lt)
        return zoomed(img, x * self.take.scale, y * self.take.scale, z)


@dataclass
class Card:
    images: list  # [(seconds into the shot, png)]: later images fade in over the earlier ones
    dur: float
    kb: tuple = (1.0, 1.04)  # slow push-in

    def image(self, lt):
        z = self.kb[0] + (self.kb[1] - self.kb[0]) * lt / self.dur
        idx = max((i for i, (t, _) in enumerate(self.images) if t <= lt), default=0)
        img = load(self.images[idx][1])
        t_in = self.images[idx][0]
        if idx and lt - t_in < 0.5:
            img = Image.blend(load(self.images[idx - 1][1]), img, (lt - t_in) / 0.5)
        return zoomed(img, img.width / 2, img.height / 2, z)


_cache = {}


def load(path):
    path = str(path)
    if path not in _cache:
        if len(_cache) > 64:
            _cache.clear()
        _cache[path] = Image.open(path).convert("RGB")
    return _cache[path]


def zoomed(img, cx, cy, z):
    if abs(z - 1) < 1e-3 and img.size == (W, H):
        return img
    w, h = img.width / z, img.height / z
    x0 = min(max(cx - w / 2, 0), img.width - w)
    y0 = min(max(cy - h / 2, 0), img.height - h)
    return img.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))


def badge(img, text):
    img = img.copy()
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=FONT)
    x1, y0 = W - 48, 44
    d.rounded_rectangle((x1 - tw - 44, y0, x1, y0 + 56), radius=28, fill=(12, 12, 16), outline=(242, 193, 78), width=2)
    d.text((x1 - tw - 22, y0 + 10), text, font=FONT, fill=(242, 193, 78))
    return img


def plan():
    """The edit, shot by shot, following demo/STORYBOARD.md. Returns (shots, narration cues)."""
    din, mine, more = Take("dinner"), Take("own"), Take("stories")
    vo = json.loads((OUT / "vo" / "lines.json").read_text())
    cards = OUT / "cards"
    shots, cues = [], []  # cues: (narration key, shot index, seconds into that shot)

    def add(shot, say=None, at=0.3):
        shots.append(shot)
        if say:
            cues.append((say, len(shots) - 1, at))
        return shot

    # 1. cold open: Frank gives up his secret in take 2, and Josh never knew
    back = din.mark("go_back")
    a = din.line("thirty-two", back)[0]
    b = din.line("you never told me", back)[1]
    add(Clip(din, a - 0.4, b + 1.2, zoom=[(0, 700, 360, 1.04), (b - a + 2, 720, 350, 1.12)], hold=0.8))
    # 2-3. title, then the problem over three story photos
    add(Card([(0, cards / "title.png")], 3.0, kb=(1.0, 1.02)))
    per = (vo["problem"] + 0.9) / 3
    add(Card([(0, cards / "problem_1.png")], per + XF), "problem", 0.3)
    add(Card([(0, cards / "problem_2.png")], per + XF))
    add(Card([(0, cards / "problem_3.png")], per + XF))
    # 4. the solution: pick the dinner story, the briefing with its cast
    home, brief = din.mark("home"), din.mark("briefing")
    add(Clip(din, home + 0.3, din.mark("scene") + 1.0, hold=5.0,
             zoom=[(0, 640, 360, 1.0), (brief - home + 0.6, 640, 360, 1.0), (brief - home + 2.6, 860, 520, 1.25)]), "solution", 0.2)
    # 5. take 1: Frank opens, the blunt line, Josh walks in, it falls apart, the ending
    scene = din.mark("scene")
    opener = din.line("food's gettin' cold", scene)
    add(Clip(din, opener[0] - 1.2, din.line("starts in", scene)[1] + 0.9, hold=0.8), "first", 1.3)
    s1 = din.mark("t1_1_sent")
    add(Clip(din, s1 - 2.0, din.line("this week", s1)[1] + 1.1, hold=0.8))
    s2 = din.mark("t1_2_sent")
    add(Clip(din, s2 - 1.8, din.line("Don't say that", s2)[1] + 0.6, hold=0.7))
    s8 = din.mark("t1_8_sent")
    add(Clip(din, s8 - 1.8, din.line("stay with me a minute", s8)[0] + 2.2, hold=0.8, badge="a few turns later"))
    end1 = din.mark("t1_ending")
    cx, cy = din.box("t1_eBack")
    add(Clip(din, end1 - 1.0, back - 0.9, hold=1.2,
             zoom=[(0, 640, 360, 1.0), (4.5, 640, 360, 1.0), (6.5, 640, 470, 1.3)]), "reveal", 1.6)
    # 6. take 2: go back, Mom tells you about the $8,000, Dad tells his story, the ending
    s3 = din.mark("t2_2_sent")
    add(Clip(din, back - 0.9, back + 1.8, zoom=[(0, cx, cy, 1.3), (1.4, 640, 360, 1.0)], hold=0.6), "back", 0.0)
    add(Clip(din, s3 - 1.8, din.line("Plans are easy", s3)[0] - 0.1, hold=0.8, badge="take 2"))
    s6 = din.mark("t2_6_sent")
    add(Clip(din, s6 - 1.8, b + 1.2, hold=0.8))
    end2 = din.mark("t2_ending")
    add(Clip(din, end2 - 0.3, end2 + 7.5, hold=1.8, zoom=[(0, 640, 360, 1.0), (4.5, 640, 360, 1.0), (7.0, 640, 440, 1.15)]))
    # 7. breadth: one sentence becomes a scene; other stories; languages
    add(Clip(mine, mine.mark("own") + 1.4, mine.mark("gen_start") + 0.6, hold=0.6), "breadth", 0.3)
    add(Clip(mine, mine.mark("gen_start") + 0.6, mine.mark("gen_done") + 0.2, speed=12.0, voices=False, hold=0.4,
             badge="about a minute, sped up"))
    add(Clip(mine, mine.mark("gen_done") + 0.2, mine.mark("gen_done") + 3.6, hold=0.8))
    for title in ("The Red-Headed League", "Sold: One Eiffel Tower", "Blind Date"):
        s = more.mark(f"story_{title}")
        add(Clip(more, s + 0.3, s + 2.9, hold=4.0))
    add(Clip(more, more.mark("languages") + 0.2, more.mark("end") - 0.2, hold=4.0))
    # 8. how it is built, the diagram lighting up with the narration
    d, at = vo["built"], 0.6  # the highlight moves at the pauses after "three seconds" and "trust grows"
    add(Card([(0, cards / "built_0.png"), (at + 1.0, cards / "built_1.png"), (at + 8.5, cards / "built_2.png"),
              (at + 13.7, cards / "built_3.png")], d + 1.7, kb=(1.0, 1.02)), "built", at)
    # 9. the end card
    add(Card([(0, cards / "end.png")], vo["cta"] + 2.4, kb=(1.0, 1.02)), "cta", 0.5)
    return shots, cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "stand-in-demo.mp4")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=1e9)
    ap.add_argument("--plan", action="store_true", help="print the shot list and stop")
    ap.add_argument("--sound-only", action="store_true", help="keep the pictures already drawn, remix the sound")
    args = ap.parse_args()
    shots, cues = plan()
    starts, t = [], 0.0
    for s in shots:
        starts.append(t)
        t += s.dur - XF
    total = t + XF
    t0, t1 = args.t0, min(args.t1, total)
    print(f"{len(shots)} shots, {total:.1f}s")
    for i, (s, st) in enumerate(zip(shots, starts)):
        print(f"  {i:2d} {st:6.1f}s  {s.dur:5.1f}s  {type(s).__name__}{' x' + str(s.speed) if getattr(s, 'speed', 1) != 1 else ''} {getattr(s, 'badge', '')}")
    vo = json.loads((OUT / "vo" / "lines.json").read_text())
    for key, i, at in cues:
        print(f"  narration {key:9} {starts[i] + at:6.1f}s - {starts[i] + at + vo[key]:6.1f}s")
    if args.plan:
        return

    # pictures
    video = OUT / "video.mp4"
    if not args.sound_only:
        draw(shots, starts, total, t0, t1, video)
    mix(shots, starts, cues, t0, t1, video, args.out)


def draw(shots, starts, total, t0, t1, video):
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "17", "-preset", "medium",
                            "-pix_fmt", "yuv420p", str(video)], stdin=subprocess.PIPE)
    n0, n1 = int(t0 * FPS), int(t1 * FPS)
    for n in range(n0, n1):
        t = n / FPS
        live = [(s, st) for s, st in zip(shots, starts) if st <= t < st + s.dur]
        s, st = live[-1]
        img = s.image(t - st)
        if getattr(s, "badge", ""):
            img = badge(img, s.badge)
        if len(live) > 1 and t - st < XF:
            p, pst = live[-2]
            prev = p.image(t - pst)
            if getattr(p, "badge", ""):
                prev = badge(prev, p.badge)
            img = Image.blend(prev, img, smooth((t - st) / XF))
        if t < 0.6:
            img = Image.blend(Image.new("RGB", (W, H)), img, smooth(t / 0.6))
        if t > total - 0.9:
            img = Image.blend(Image.new("RGB", (W, H)), img, smooth((total - t) / 0.9))
        enc.stdin.write(img.tobytes())
        if n % (FPS * 10) == 0:
            print(f"  frame {t:.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()


def mix(shots, starts, cues, t0, t1, video, out):
    """The voices as the page played them, the narration on top, voices ducked under it."""
    inputs, f, k = [], [], 0
    narr = []
    for key, i, at in cues:
        st = starts[i] + at - t0
        dur = json.loads((OUT / "vo" / "lines.json").read_text())[key]
        narr.append((st, st + dur))
        inputs += ["-i", str(OUT / "vo" / f"{key}.mp3")]
        f.append(f"[{k + 1}:a]adelay={max(int(st * 1000), 0)}:all=1,volume=1.15[n{k}]")
        k += 1
    nk = k
    for s, st in zip(shots, starts):
        if not isinstance(s, Clip) or not s.voices:
            continue
        for v0, v1, path, rate, pos in s.take.voices:
            lt = s.local(v0)
            if lt is None or not (s.a <= v0 < s.b):
                continue
            at = st + lt - t0
            if at < 0 or at > t1 - t0:
                continue
            inputs += ["-i", str(path)]
            f.append(f"[{k + 1}:a]atrim=0:{pos:.3f},asetpts=PTS-STARTPTS,atempo={rate},adelay={int(at * 1000)}:all=1[c{k}]")
            k += 1
    length = t1 - t0
    duck = ",".join(f"volume=enable='between(t,{a:.2f},{b:.2f})':volume=0.32" for a, b in narr) or "anull"
    chars = "".join(f"[c{i}]" for i in range(nk, k))
    if k > nk:
        f.append(f"{chars}amix=inputs={k - nk}:normalize=0:dropout_transition=0,{duck}[chars]")
    else:
        f.append(f"anullsrc=r=44100:cl=stereo,atrim=0:{length:.3f}[chars]")
    f.append("".join(f"[n{i}]" for i in range(nk)) + f"[chars]amix=inputs={nk + 1}:normalize=0:dropout_transition=0,"
             f"apad,atrim=0:{length:.3f},afade=t=out:st={max(length - 0.9, 0):.3f}:d=0.9,"
             f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]")
    (OUT / "mix.txt").write_text(";\n".join(f))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), *inputs, "-filter_complex_script", str(OUT / "mix.txt"),
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    "-shortest", str(out)], check=True)
    print(f"{out}: {length:.1f}s")


if __name__ == "__main__":
    main()
