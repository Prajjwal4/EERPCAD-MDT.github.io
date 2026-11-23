#!/usr/bin/env python3
"""Scene S01 - stickman, dark room, red light.   Narration (unchanged):
    "Your mind has weaknesses you probably don't even know about."

Story beat: the stickman taps his temple ("your mind"); on the word 'weaknesses' red cracks
spread across his head; he doesn't notice - he looks around and shrugs - while his shadow on the
wall behind him starts acting on its own and turns to look at him.

Everything is drawn in code (NumPy + Pillow); FFmpeg encodes. No external generation services.
Outputs: scenes/S01.mp4, assets/visuals/visual_01.png (keyframe), qa/S01_preview.png.
Caption windows and every beat are timed to the real voiceover (assets/audio/voiceover_timing.json).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, FFmpegWriter, Scene, SceneRenderer
from stickman import Style, Timeline, render_character, skeleton_bounds, smooth

cfg = load_config()
st = Style.from_cfg(cfg)
S = st.world_scale                                   # final px -> world px
WW, WH = round(cfg.width * S), round(cfg.height * S)  # world canvas 1296 x 2304
FLOOR = 1340.0                                       # floor line (final px)
HIP_X = 520.0
REGION = (150, 330, 960, 1440)                       # character draw region (final px)
LAMP = (160.0, 60.0)                                 # bulb position (final px)
AXIS = (760.0, 930.0)                                # where the light beam points
RED = np.asarray(cfg.rgb01("accent_red"), np.float32)
BLUE = np.asarray(cfg.rgb01("research_blue"), np.float32)
GREY = np.asarray(cfg.rgb01("secondary_text"), np.float32)
WHITE = np.asarray(cfg.rgb01("primary_text"), np.float32)

NARRATION = "Your mind has weaknesses you probably don't even know about."
HIGHLIGHT = ("weaknesses",)


def load_timing() -> list[dict]:
    """Word timings of scene 1 from the real voiceover (assets/audio/voiceover_timing.json)."""
    d = json.load(open(cfg.path("assets/audio/voiceover_timing.json")))
    line = d["lines"][0]
    assert line["scene"] == 1
    shown = NARRATION.split()
    assert len(line["words"]) == len(shown), "timing words must map 1:1 onto the approved narration"
    return [{"shown": w, **t} for w, t in zip(shown, line["words"])]


WORDS = load_timing()
FR = {t["word"]: (t["start_frame"], t["end_frame"]) for t in WORDS}          # spoken frames per word
HIT = FR["weaknesses"][0]            # frame the word 'weaknesses' starts being spoken (29)
SPEECH_END = WORDS[-1]["end_frame"]  # last word ends (110)
# 10 words > 7-word limit -> two caption groups: split before "you".
_split = 4
GROUPS = [(" ".join(t["shown"] for t in WORDS[:_split]), WORDS[0]["start_frame"], WORDS[_split]["start_frame"] + 1,
           [t["start_frame"] for t in WORDS[:_split]]),
          (" ".join(t["shown"] for t in WORDS[_split:]), WORDS[_split]["start_frame"] + 1, SPEECH_END + 14,
           [WORDS[_split]["start_frame"] + 1] + [t["start_frame"] for t in WORDS[_split + 1:]])]


# ================================================================= timeline
def make_timeline() -> Timeline:
    """Beats are anchored to spoken words: 'mind' f10 (temple tap) -> 'weaknesses' f29 (flinch, cracks) ->
    'you probably' f51 (looks around) -> 'don't even know' f74 (shrug) -> 'about' f96 -> silence f110."""
    hang = lambda s: (34.0 * s, 232.0)
    tap_hi, tap_lo = (29.0, -96.0), (27.0, -82.0)
    return Timeline({
        "lean": [(0, 3), (10, 0), (HIT, 0), (HIT + 3, -3), (HIT + 10, 0), (96, 0), (118, 2)],
        "head": [(0, 12), (9, 0), (14, -4), (HIT - 1, -4), (HIT + 2, -12), (HIT + 9, 0), (54, -8), (62, 8), (72, 0),
                 (90, 10), (108, 4), (130, 0)],
        "handR": [(0, hang(1)), (8, hang(1)), (14, tap_hi), (17, tap_lo), (20, tap_hi), (23, tap_lo), (26, tap_hi),
                  (HIT - 1, tap_hi), (HIT + 13, hang(1)), (64, hang(1)), (76, (138.0, 48.0)), (94, (138.0, 44.0)),
                  (108, hang(1))],
        "handL": [(0, hang(-1)), (64, hang(-1)), (76, (-138.0, 48.0)), (94, (-138.0, 44.0)), (108, hang(-1))],
        "shrug": [(0, 0), (66, 0), (78, 1), (94, 1), (108, 0)],
        "gaze": [(0, (0.0, 0.9)), (4, (0.0, 0.9)), (10, (0.0, 0.0)), (HIT, (0.0, 0.0)), (HIT + 9, (0.5, -0.2)),
                 (52, (-1.0, 0.0)), (60, (1.0, 0.0)), (68, (0.0, -0.3)), (92, (0.0, -0.4)), (108, (0.0, 0.2)),
                 (122, (0.0, 0.0))],
        "brow": [(0, 0.0), (12, 0.3), (HIT - 1, 0.3), (HIT + 2, 1.0), (100, 1.0), (125, 0.6)],
        "raise_": [(0, 0.0), (12, 0.6), (HIT - 1, 0.6), (HIT + 2, 1.0), (HIT + 18, 0.3), (125, 0.2)],
        "wide": [(0, 0.0), (HIT - 1, 0.0), (HIT + 2, 1.0), (HIT + 16, 0.3), (125, 0.2)],
        "mouth": [(0, "flat"), (12, "o"), (HIT - 1, "o"), (HIT + 9, "wavy"), (88, "frown"), (118, "flat")],
    })


TL = make_timeline()


def blink(f: float) -> float:
    return max((max(0.0, 1 - abs(f - c) / 2.0) for c in (8, 46, 104)), default=0.0)


def figure_pose(f: float) -> dict:
    p = TL.at(f)
    p["hip_x"] = HIP_X + 1.5 * math.sin(f * 0.07)          # slight idle sway
    p["blink"] = blink(f)
    return p


def crack_progress(f: float) -> float:
    return smooth((f - HIT) / 22.0)


def shadow_pose(f: float) -> dict:
    """The wall shadow copies him (4 frames late), then during 'probably don't even know' acts on its own:
    it turns its head toward him and reaches for his head - and slips back into copying him in the silence."""
    p = figure_pose(f - 4)
    w = smooth((f - 58) / 24.0) * (1 - smooth((f - 114) / 20.0))
    p["head"] = p["head"] + (-26.0 - p["head"]) * w
    p["lean"] = p["lean"] + (-8.0 - p["lean"]) * w
    hl = p["handL"]
    p["handL"] = (hl[0] + (-125.0 - hl[0]) * w, hl[1] + (-150.0 - hl[1]) * w)
    return p


def light_k(f: float) -> float:
    """Red light level: fades up over 20 frames, gentle flicker, a pulse when the cracks appear."""
    k = smooth(f / 12.0) * (0.92 + 0.06 * math.sin(f * 0.21) + 0.03 * math.sin(f * 0.57 + 1.0))
    return k * (1 + 0.25 * math.exp(-(((f - (HIT + 2)) / 3.0) ** 2)))


# ============================================================== static world
def fbm(h: int, w: int, octaves: int, seed: int, base: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        grid = rng.random((max(2, int(n * h / w) + 2), n + 2)).astype(np.float32)
        acc += amp * np.asarray(Image.fromarray(grid, "F").resize((w, h), Image.BICUBIC), np.float32)
        tot += amp; amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)


class World:
    """Everything that does not change per frame, precomputed once at world resolution."""

    def __init__(self):
        X = (np.arange(WW, dtype=np.float32) / S)[None, :].repeat(WH, 0)
        Y = (np.arange(WH, dtype=np.float32) / S)[:, None].repeat(WW, 1)
        self.X, self.Y = X, Y
        # ---- structure lines drawn exactly with Pillow
        seams = Image.new("L", (WW, WH), 0)
        d = ImageDraw.Draw(seams)
        for x in range(0, 1081, 180):                       # wall panel seams
            d.line([(x * S, 0), (x * S, FLOOR * S)], fill=255, width=3)
        d.line([(0, 1090 * S), (WW, 1090 * S)], fill=255, width=3)      # wainscot rail
        vp = (540 * S, 1010 * S)
        boards = Image.new("L", (WW, WH), 0)
        db = ImageDraw.Draw(boards)
        for k in range(-9, 10):                             # floor boards converging to a vanishing point
            db.line([vp, (540 * S + k * 330 * S, WH)], fill=255, width=2)
        db.rectangle([0, 0, WW, FLOOR * S], fill=0)         # ...but only on the floor
        trim = Image.new("L", (WW, WH), 0)
        dt = ImageDraw.Draw(trim)
        dt.rectangle([858 * S, 455 * S, 1012 * S, FLOOR * S], outline=255, width=5)  # door frame
        dt.line([(0, FLOOR * S), (WW, FLOOR * S)], fill=255, width=4)                 # baseboard
        seams_a = np.asarray(seams.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        boards_a = np.asarray(boards.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        trim_a = np.asarray(trim.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        tex = fbm(WH, WW, 5, 3)
        wall = 0.115 + 0.05 * (tex - 0.5)
        wall = np.where(Y > 1090, wall * 0.82, wall)
        wall *= (1 - 0.55 * seams_a)
        door = (X > 858) & (X < 1012) & (Y > 455)
        wall = np.where(door, wall * 0.42, wall)
        floor = (0.085 + 0.03 * (tex - 0.5)) * (1 - 0.4 * smooth_np((Y - FLOOR) / 560)) * (1 - 0.6 * boards_a)
        alb = np.where(Y < FLOOR, wall, floor) + 0.05 * trim_a
        self.alb = alb.astype(np.float32)

        # ---- red light: cone from the hanging bulb + soft halo around it
        lx, ly = LAMP
        dx, dy = X - lx, Y - ly
        dist = np.sqrt(dx * dx + dy * dy) + 1e-3
        axis_ang = math.atan2(AXIS[1] - ly, AXIS[0] - lx)
        th = np.arctan2(dy, dx) - axis_ang
        th = (th + np.pi) % (2 * np.pi) - np.pi
        cone = np.exp(-((th / 0.46) ** 2)) / (1 + (dist / 950) ** 2)
        halo = np.exp(-((dist / 330) ** 2))
        self.cone = np.clip(cone * 1.9, 0, 1.5).astype(np.float32)
        gloss = np.exp(-((((X - 700) / 300) ** 2) + (((Y - 1470) / 85) ** 2)))     # light pool on the floor
        I = self.cone + 0.55 * halo + 0.9 * gloss * (Y > FLOOR)
        self.dark = (self.alb[None] * (BLUE * 0.30)[:, None, None]).astype(np.float32)
        self.lit = (self.alb[None] * RED[:, None, None] * (I[None] * 5.0)).astype(np.float32)
        # ---- bulb (glow sprite + cord)
        core = np.exp(-((dist / 11) ** 2))
        bulb = core[None] * np.asarray([1.0, 0.78, 0.78], np.float32)[:, None, None] * 1.6
        bulb += (np.exp(-((dist / 75) ** 2)) * 0.5)[None] * RED[:, None, None]
        self.bulb = bulb.astype(np.float32)
        cord = Image.new("L", (WW, WH), 0)
        ImageDraw.Draw(cord).line([(lx * S, 0), (lx * S, (ly - 14) * S)], fill=255, width=3)
        self.dark += (np.asarray(cord, np.float32) / 255 * 0.05)[None]
        # ---- volumetric haze texture (scrolls) lit by the beam
        self.haze_tex = fbm(WH + 220, WW + 420, 4, 9, base=4) ** 1.7
        self.beam = (np.clip(self.cone, 0, 1) * 0.9 + 0.7 * halo).astype(np.float32)
        # ---- dust particles
        rng = np.random.default_rng(5)
        n = 90
        self.px = rng.uniform(0, cfg.width, n); self.py = rng.uniform(150, 1400, n)
        self.pvx = rng.uniform(0.08, 0.45, n); self.pvy = rng.uniform(-0.45, -0.08, n)
        self.psize = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]); self.pph = rng.uniform(0, 6.28, n)
        self.sprites = []
        for r in (1.4, 2.2, 3.4):
            yy, xx = np.mgrid[-6:7, -6:7].astype(np.float32)
            self.sprites.append(np.exp(-(xx * xx + yy * yy) / (2 * (r * S * 0.5) ** 2)).astype(np.float32))

    def dust(self, f: float) -> np.ndarray:
        P = np.zeros((WH, WW), np.float32)
        x = (self.px + self.pvx * f) % cfg.width
        y = self.py + self.pvy * f
        for i in range(len(x)):
            wx, wy = int(x[i] * S), int(y[i] * S)
            if not (8 <= wx < WW - 8 and 8 <= wy < WH - 8):
                continue
            b = (0.10 + 1.1 * float(self.cone[wy, wx])) * (0.6 + 0.4 * math.sin(self.pph[i] + f * 0.2))
            P[wy - 6:wy + 7, wx - 6:wx + 7] += self.sprites[self.psize[i]] * b
        return P


def smooth_np(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ================================================================ per frame
def _paste(mask: np.ndarray, x0: int, y0: int) -> Image.Image:
    full = Image.new("L", (WW, WH), 0)
    full.paste(Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L"), (x0, y0))
    return full


def _affine(img: Image.Image, coef) -> np.ndarray:
    return np.asarray(img.transform(img.size, Image.AFFINE, coef, Image.BILINEAR), np.float32) / 255


FLW = FLOOR * S
WALL_M, WALL_DX = 1.32, 270.0
AXW, AXW2 = HIP_X * S, (HIP_X + WALL_DX) * S
WALL_COEF = (1 / WALL_M, 0, AXW - AXW2 / WALL_M, 0, 1 / WALL_M, FLW - FLW / WALL_M)
FA, FB = 0.28, 0.9                                                     # floor-shadow squash / shear
FLOOR_COEF = (1, -FB / FA, (FB / FA) * FLW, 0, -1 / FA, FLW * (1 + 1 / FA))


def render_world(world: World, f: float, with_char: bool = True) -> np.ndarray:
    k = light_k(f)
    frame = world.dark.copy()
    fig = figure_pose(f)
    L = render_character(fig, st, REGION, FLOOR, crack=crack_progress(f))
    sh = render_character(shadow_pose(f), st, REGION, FLOOR, silhouette_only=True, crack=crack_progress(f - 8))
    # ---- cast shadows (wall: big, offset, from the lagging/independent pose; floor: from his real pose)
    sil_wall = _paste(sh.body, sh.x0, sh.y0)
    sil_floor = _paste(L.body, L.x0, L.y0)
    wall_sh = np.asarray(Image.fromarray((_affine(sil_wall, WALL_COEF) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
    floor_sh = np.asarray(Image.fromarray((_affine(sil_floor, FLOOR_COEF) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
    shadow = np.maximum(wall_sh, floor_sh * 0.9)
    unlit = 1 - 0.93 * shadow
    frame += world.lit * (unlit * k)[None]
    # ---- volumetric haze drifting through the beam
    ox, oy = int(f * 1.6), int(f * 0.5)
    haze = world.haze_tex[oy:oy + WH, ox:ox + WW] * world.beam * unlit
    frame += (haze * 0.38 * k)[None] * RED[:, None, None]
    frame += world.bulb * (k * (1 + 0.05 * math.sin(f * 1.3)))
    # ---- dust motes (warm-white in the beam)
    frame += world.dust(f)[None] * (np.asarray([1.0, 0.62, 0.62], np.float32)[:, None, None] * 0.55 * k)
    # ---- cracks projected onto the wall shadow: they glow on the wall behind him
    if sh.crack is not None:
        cw = _affine(_paste(sh.crack, sh.x0, sh.y0), WALL_COEF)
        glow = np.asarray(Image.fromarray((cw * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(7)), np.float32) / 255
        frame += (cw * 1.2 + glow * 1.4)[None] * RED[:, None, None] * k
    if not with_char:
        return frame

    # ---- the stickman
    x0, y0 = L.x0, L.y0
    h, w = L.body.shape
    reg = frame[:, y0:y0 + h, x0:x0 + w]
    ol, bd = L.outline, L.body
    reg_o = GREY * 0.42
    reg[:] = reg * (1 - ol[None]) + reg_o[:, None, None] * ol[None]            # light outline
    reg *= (1 - bd[None])
    reg += bd[None] * 0.012                                                    # black body
    rim = np.clip(ol - np.roll(ol, 9, axis=1), 0, 1)                           # red rim on the lamp side
    rim = np.asarray(Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255
    reg += (rim * 1.35 * k)[None] * RED[:, None, None]
    # reflection of the outline on the floor
    fr = round(FLW) - y0
    d = min(h - fr, 110)
    if d > 0 and fr - d > 0:
        refl = ol[fr - d:fr][::-1]
        grad = np.linspace(0.45, 0.0, d, dtype=np.float32)[:, None]
        reg[:, fr:fr + d] += (refl * grad)[None] * (RED * 0.5 + GREY * 0.12)[:, None, None] * k
    # face
    reg[:] = reg * (1 - 0.96 * L.face_w[None]) + WHITE[:, None, None] * (0.96 * L.face_w[None])
    reg *= (1 - 0.98 * L.face_k[None])
    # cracks on his head: hot red core + glow
    if L.crack is not None:
        pulse = 0.85 + 0.15 * math.sin(f * 0.35)
        cb = np.asarray(Image.fromarray((L.crack * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
        reg += (L.crack * 1.5 + cb * 1.1)[None] * RED[:, None, None] * pulse
        reg += (L.crack * 0.35)[None] * WHITE[:, None, None]
    return frame


def world_image(world: World, f: float) -> Image.Image:
    fr = render_world(world, f)
    return Image.fromarray((np.clip(fr, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8), "RGB")


# ===================================================================== build
def build_scene(world: World) -> Scene:
    assert " ".join(g for g, *_ in GROUPS) == NARRATION, "captions must equal the approved narration"
    caps = [CaptionEvent(CaptionGroup(t, HIGHLIGHT), s, e, kind="caption", position="lower", word_starts=ws)
            for t, s, e, ws in GROUPS]
    return Scene(source=lambda f, n: world_image(world, f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.08, ease="linear"))      # slow 100% -> 108%


def check_layout() -> dict:
    """Every pose across the scene must keep the stickman inside the safe area, clear of the captions."""
    sx0, sy0, sx1, sy1 = 1e9, 1e9, -1e9, -1e9
    for f in range(150):
        x0, y0, x1, y1 = skeleton_bounds(figure_pose(f), st, FLOOR)
        sx0, sy0, sx1, sy1 = min(sx0, x0), min(sy0, y0), max(sx1, x1), max(sy1, y1)
    l, t, r, b = cfg.safe_rect
    cap_top = 1447
    ok = sx0 >= l and sx1 <= r and sy0 >= t and sy1 <= cap_top
    return {"bounds": [round(v) for v in (sx0, sy0, sx1, sy1)], "safe": [l, t, r, cap_top], "ok": ok}


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return {**j["streams"][0], "format_duration": float(j["format"]["duration"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=int, help="only render these world frames to qa/s01_dev_*.png and exit")
    a = ap.parse_args()
    world = World()
    R = SceneRenderer(cfg)
    if a.stills is not None:
        for f in a.stills:
            img = world_image(world, f)
            x = R._planar(np.asarray(img.resize((cfg.width, cfg.height), Image.LANCZOS), np.float32))
            x = R.grade.apply(x, f)
            Image.fromarray((np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(cfg.path(f"qa/s01_dev_{f:03d}.png"))
        return
    lay = check_layout()
    print("layout:", lay)
    if not lay["ok"]:
        sys.exit("stickman leaves the safe area")
    scene = build_scene(world)
    out, preview, key = cfg.path("scenes/S01.mp4"), cfg.path("qa/S01_preview.png"), cfg.path("assets/visuals/visual_01.png")
    n = R.render_scene_files(scene, out, cfg.path("scenes/clean/S01_clean.mp4"))      # strict: raises unless exactly 150 frames
    p = ffprobe(out)
    print(json.dumps(p, indent=2))
    ok = (n == 150 and int(p["nb_read_frames"]) == 150 and (p["width"], p["height"]) == (1080, 1920)
          and p["r_frame_rate"] == "30/1" and abs(p["format_duration"] - 5.0) < 0.01)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf",
                    f"select=eq(n\\,{HIT + 11})", "-frames:v", "1", str(preview)], check=True)
    # Keyframe still (no captions, zoom 1.0, final grade) = visual_01.png, native 1080x1920.
    img = world_image(world, HIT + 24)
    x = R.grade.apply(R._planar(np.asarray(img.resize((cfg.width, cfg.height), Image.LANCZOS), np.float32)), 60)
    Image.fromarray((np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(key)
    # QA only: the scene with its 5 s slice of the real voiceover, to check sync by eye/ear (S01.mp4 itself stays silent).
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-ss", "0", "-t", "5", "-i",
                    str(cfg.path("assets/audio/voiceover_01.wav")), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", str(cfg.path("qa/S01_with_voice.mp4"))], check=True)
    print("preview:", Image.open(preview).size, "| keyframe:", Image.open(key).size)
    print("S01 VERIFIED" if ok else "S01 FAILED VERIFICATION")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
