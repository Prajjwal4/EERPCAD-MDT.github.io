#!/usr/bin/env python3
"""Scene S02 - SOCIAL EXCLUSION.   "First, social exclusion: being ignored can threaten our sense of belonging."

A lone stickman walks up to a group of four and waves. They glance at him ("exclusion"), then turn their backs
one by one, huddle and laugh, and walk away into the dark ("our sense of belonging") while the wide red beam that
lit the group dies and a narrow red spotlight finds him alone.

Timing is read from the approved script + real voiceover (script_plan.py): the line starts 5.15 s, ends 10.62 s,
so "belonging." (f147-168) crosses the 150-frame scene end; the last caption group carries on into S03.
Outputs: scenes/S02.mp4, assets/visuals/visual_02.png, qa/S02_preview.png, qa/S02_with_voice.mp4, qa/S02_contact.png
"""
from __future__ import annotations

import argparse
import functools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa_scene
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, FFmpegWriter, Scene, SceneRenderer
from script_plan import load_scene
from stage import Lamp, Stage
from stickman import Style, Timeline, render_character, skeleton_bounds, smooth

cfg = load_config()
ST = Style.from_cfg(cfg)
S = ST.world_scale
WW, WH = round(cfg.width * S), round(cfg.height * S)
FLOOR = 1340.0
GREY = np.asarray(cfg.rgb01("secondary_text"), np.float32)
WHITE = np.asarray(cfg.rgb01("primary_text"), np.float32)
RED = np.asarray(cfg.rgb01("accent_red"), np.float32)

PLAN = load_scene(2)
assert PLAN.id == "S02" and PLAN.narration.startswith("First, social exclusion")
FR = {w["word"].strip(",:.").lower(): (w["start"], w["end"]) for w in PLAN.words}      # scene-local spoken frames
A_SOCIAL, A_EXCL, A_BEING, A_CAN, A_OUR, A_BELONG = (FR[k][0] for k in ("social", "exclusion", "being", "can", "our", "belonging"))

LAMP_GROUP = Lamp(pos=(930, 60), axis=(780, 960), sigma=0.50, gain=1.9)                      # wide beam on the crowd
LAMP_SOLO = Lamp(pos=(300, 40), axis=(320, 1000), sigma=0.20, gain=2.6, reach=1100, halo=260)  # narrow spotlight on him


# ================================================================ choreography
def lerp(a, b, t):
    return a + (b - a) * t


def lerp2(a, b, t):
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


HANG_L, HANG_R = (-34.0, 232.0), (34.0, 232.0)


def blink(f, centers):
    return max((max(0.0, 1 - abs(f - c) / 2.0) for c in centers), default=0.0)


# ---- the isolated character (P) -------------------------------------------------------------
P_TL = Timeline({
    "hip_x": [(0, 200), (8, 200), (34, 290), (90, 285), (106, 312), (150, 312)],
    "turn": [(0, 45), (8, 72), (34, 72), (42, 38), (150, 34)],
    "head": [(0, 8), (22, 0), (A_EXCL + 22, 4), (A_BEING + 10, 10), (A_CAN + 6, 16), (A_OUR + 12, 12), (150, 18)],
    "gaze": [(0, (0.9, 0.3)), (20, (1.0, 0.0)), (A_CAN, (1.0, 0.1)), (A_OUR, (0.7, 0.5)), (150, (0.3, 0.9))],
    "brow": [(0, 0.0), (A_SOCIAL, 0.0), (A_EXCL + 14, 0.5), (A_BEING + 6, 1.0), (150, 1.0)],
    "raise_": [(0, 0.0), (A_SOCIAL - 2, 0.9), (A_EXCL + 14, 0.6), (A_BEING + 6, 0.2), (150, 0.0)],
    "mouth": [(0, "flat"), (A_SOCIAL - 4, "smile"), (A_EXCL + 22, "wavy"), (A_BEING + 6, "frown"), (150, "frown")],
    "lean": [(0, 2), (34, 0), (A_CAN, 0), (A_OUR + 10, 5), (150, 7)],
    "wide": [(0, 0.0), (A_EXCL + 14, 0.6), (A_BEING + 10, 0.3), (150, 0.1)],
})


def p_pose(f: float) -> dict:
    p = P_TL.at(f)
    rw = smooth((f - (A_SOCIAL + 2)) / 6.0) * (1 - smooth((f - (A_EXCL + 24)) / 40.0))     # hand-wave weight
    wave = (58.0 + 15.0 * math.sin(0.55 * (f - A_SOCIAL)) * (1 - 0.6 * smooth((f - A_EXCL) / 30.0)),
            -140.0 + 18.0 * math.cos(0.55 * (f - A_SOCIAL)))
    p["handR"] = lerp2(HANG_R, wave, rw)
    reach = smooth((f - 90) / 10.0) * (1 - smooth((f - 112) / 14.0))                       # reaches toward them, then drops
    p["handL"] = lerp2(HANG_L, (70.0, 40.0), reach)
    # two short walks: toward the group (f8-34) and one more hopeful step (f90-108)
    for lo, hi in ((8, 36), (90, 110)):
        if lo <= f <= hi:
            ww = smooth((f - lo) / 4.0) * (1 - smooth((f - (hi - 6)) / 6.0))
            ph = (f - lo) * 0.62
            p["footL"] = (-26 + 58 * ww * math.sin(ph), 347 - 34 * ww * max(0.0, math.cos(ph)))
            p["footR"] = (26 - 58 * ww * math.sin(ph), 347 - 34 * ww * max(0.0, -math.cos(ph)))
            sw = 34 * ww * math.sin(ph)
            p["handL"] = (p["handL"][0] + sw, p["handL"][1])
            if rw < 0.5:
                p["handR"] = (p["handR"][0] - sw, p["handR"][1])
    p["blink"] = blink(f, (14, 62, 118))
    return p


# ---- the group (G1..G4); turn: negative = toward P (screen-left), ~+100 = back turned to him ----------------
GROUP = [  # x, floor, scale, phase, turn track, walk start frame
    dict(x=590, fl=1335, sc=0.97, ph=0.0, walk=A_CAN + 2,
         turn=[(0, -30), (A_SOCIAL, -30), (A_EXCL - 2, -55), (A_EXCL + 10, -55), (A_EXCL + 34, 105)]),
    dict(x=705, fl=1318, sc=0.90, ph=1.7, walk=A_CAN + 8,
         turn=[(0, -5), (A_SOCIAL, -5), (A_EXCL + 2, -40), (A_EXCL + 16, -40), (A_EXCL + 40, 110)]),
    dict(x=830, fl=1338, sc=1.00, ph=3.1, walk=A_CAN + 14,
         turn=[(0, 25), (A_SOCIAL + 6, 25), (A_EXCL + 8, -30), (A_EXCL + 22, -30), (A_EXCL + 46, 110)]),
    dict(x=940, fl=1322, sc=0.92, ph=4.4, walk=A_CAN + 20,
         turn=[(0, -20), (A_SOCIAL + 10, -20), (A_EXCL + 14, -35), (A_EXCL + 28, -35), (A_EXCL + 52, 105)]),
]
for g in GROUP:
    g["tl"] = Timeline({"turn": g["turn"] + [(g["walk"], 105), (g["walk"] + 8, 108)]})
SPEED, DEPTH_Y, DEPTH_S = 6.5, 0.52, 0.0016


@functools.lru_cache(maxsize=None)
def walked(ws: int, f_int: int) -> float:
    return sum(SPEED * smooth((t - ws) / 10.0) for t in range(ws, f_int + 1)) if f_int >= ws else 0.0


def member(i: int, f: float) -> tuple[dict, float, float]:
    g = GROUP[i]
    ph, ws = g["ph"], g["walk"]
    p = {"turn": g["tl"].at(f)["turn"]}
    dist = walked(ws, int(f)) + (SPEED * 0.5 * smooth((f - int(f)) + 0)) * (f >= ws) * 0.0
    x, fl, sc = g["x"] + dist, g["fl"] - DEPTH_Y * dist, g["sc"] - DEPTH_S * dist
    # chatting
    p["handR"] = (32 + 8 * math.sin(0.33 * f + ph), 182 - 24 * max(0.0, math.sin(0.21 * f + ph * 1.7)))
    p["handL"] = (-28 - 5 * math.sin(0.27 * f + ph), 214)
    p["head"] = 4 * math.sin(0.2 * f + ph)
    p["lean"] = 1.5 * math.sin(0.17 * f + ph)
    p["mouth"] = "o" if math.sin(0.8 * f + ph) > 0.4 else "smile"
    p["blink"] = blink(f, (20 + 9 * i, 77 + 5 * i))
    p["gaze"] = (0.0, 0.0)
    laugh = smooth((f - (A_EXCL + 44)) / 8.0) * (1 - smooth((f - (A_CAN + 8)) / 8.0))   # huddled laughter after the turn
    p["lean"] += laugh * 3.2 * math.sin(0.95 * f + ph)
    p["shrug"] = laugh * (0.5 + 0.5 * math.sin(0.95 * f + ph))
    if f >= ws:                                                                          # walks away (profile, facing right)
        w = smooth((f - ws) / 6.0)
        phi = (f - ws) * 0.62 + ph
        p["footL"] = (-26 + 56 * w * math.sin(phi), 347 - 32 * w * max(0.0, math.cos(phi)))
        p["footR"] = (26 - 56 * w * math.sin(phi), 347 - 32 * w * max(0.0, -math.cos(phi)))
        p["handL"] = (-20 - 38 * w * math.sin(phi), 222)
        p["handR"] = (20 + 38 * w * math.sin(phi), 222)
        p["lean"] = 2.0
    p["hip_x"] = x
    return p, fl, sc


def k_wide(f: float) -> float:       # beam on the group: up fast, then dies as they turn their backs
    k = smooth(f / 12.0) * (0.93 + 0.05 * math.sin(f * 0.21) + 0.03 * math.sin(f * 0.57 + 1.0))
    return k * (1 - 0.94 * smooth((f - (A_BEING + 6)) / 34.0))


def k_solo(f: float) -> float:       # narrow spotlight on him: a faint hint at first, then it finds him
    return (0.16 * smooth((f - 3) / 14.0) + 1.05 * smooth((f - (A_BEING + 2)) / 36.0)) * (0.97 + 0.03 * math.sin(f * 0.43))


# ================================================================== rendering
STAGE = None


def _paste(mask: np.ndarray, x0: int, y0: int) -> Image.Image:
    full = Image.new("L", (WW, WH), 0)
    full.paste(Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L"), (x0, y0))
    return full


def _affine(img: Image.Image, coef) -> np.ndarray:
    return np.asarray(img.transform(img.size, Image.AFFINE, coef, Image.BILINEAR), np.float32) / 255


def _blur(a: np.ndarray, r: float) -> np.ndarray:
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255


def _clip(x0: int, y0: int, h: int, w: int):
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, WW), min(y0 + h, WH)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    lx, ly = fx0 - x0, fy0 - y0
    return (slice(fy0, fy1), slice(fx0, fx1)), (slice(ly, ly + fy1 - fy0), slice(lx, lx + fx1 - fx0))


def all_actors(f: float):
    out = [("P", *(lambda p: (p, FLOOR, 1.0))(p_pose(f)))]
    for i in range(4):
        out.append((f"G{i + 1}", *member(i, f)))
    return sorted(out, key=lambda a: a[2])           # far (high on the floor) first


def render_world(f: float) -> np.ndarray:
    kw, ks = k_wide(f), k_solo(f)
    ks_list = [kw, ks]
    lamp_x = (kw * LAMP_GROUP.pos[0] + ks * LAMP_SOLO.pos[0]) / max(kw + ks, 1e-3)
    acts = []
    for name, pose, fl, sc in all_actors(f):
        st = ST.scaled(sc)
        hx = pose["hip_x"]
        region = (int(hx - 330 * sc), int(fl - 860 * sc), int(hx + 330 * sc), int(fl + 120))
        acts.append((name, pose, fl, sc, hx, render_character(pose, st, region, fl)))

    # ---- shadows: floor shadow for everyone (away from the dominant lamp); big wall shadow for P only
    floor_sh = np.zeros((WH, WW), np.float32)
    wall_sh = np.zeros((WH, WW), np.float32)
    for name, pose, fl, sc, hx, L in acts:
        h, w = L.body.shape
        FLW = fl * S
        b = float(np.clip((hx - lamp_x) / 700.0, -0.9, 0.9))
        FA = 0.28
        coef = (1, -b / FA, (b / FA) * FLW, 0, -1 / FA, FLW * (1 + 1 / FA))
        view = _clip(L.x0, L.y0, h, w)
        if view is None:
            continue
        full = Image.new("L", (WW, WH), 0)
        full.paste(Image.fromarray((np.clip(L.body, 0, 1) * 255).astype(np.uint8), "L"), (L.x0, L.y0))
        floor_sh = np.maximum(floor_sh, _affine(full, coef) * 0.9)
        if name == "P":
            sgn = 1.0 if hx >= lamp_x else -1.0
            m, dx = 1.28, sgn * float(np.clip(abs(hx - lamp_x) * 0.40, 170, 230))
            axw, axw2 = hx * S, (hx + dx) * S
            wcoef = (1 / m, 0, axw - axw2 / m, 0, 1 / m, FLW - FLW / m)
            wall_sh = _affine(full, wcoef)
    shadow = np.maximum(_blur(wall_sh, 5), _blur(floor_sh, 4))
    unlit = 1 - 0.93 * shadow
    frame = STAGE.compose(f, ks_list, unlit)

    # ---- characters, far to near
    for name, pose, fl, sc, hx, L in acts:
        h, w = L.body.shape
        v = _clip(L.x0, L.y0, h, w)
        if v is None:
            continue
        (fy, fx), (ly, lx) = v
        reg = frame[:, fy, fx]
        ol, bd = L.outline[ly, lx], L.body[ly, lx]
        vis = 0.30 + 0.70 * min(1.0, (kw if name != "P" else max(kw * 0.5, ks)))
        reg[:] = reg * (1 - ol[None]) + (GREY * 0.42 * vis)[:, None, None] * ol[None]
        reg *= (1 - bd[None])
        reg += bd[None] * 0.012
        if name == "P":   # lit from above by the spotlight (+ a little from the crowd's beam)
            rim = np.clip(ol - np.roll(ol, 9, axis=0), 0, 1) * 1.25 * ks + np.clip(ol - np.roll(ol, -9, axis=1), 0, 1) * 0.8 * kw
        else:             # lit from the right by the wide beam
            rim = np.clip(ol - np.roll(ol, -9, axis=1), 0, 1) * 1.35 * kw
        reg += (_blur(rim, 1.2))[None] * RED[:, None, None]
        fr_row = round(fl * S) - L.y0
        d = min(h - fr_row, 110)
        if d > 0 and fr_row - d > 0:
            refl = ol[fr_row - d:fr_row][::-1]
            grad = np.linspace(0.45, 0.0, d, dtype=np.float32)[:, None]
            kk = (ks if name == "P" else kw)
            xs = slice(0, reg.shape[2])
            reg[:, fr_row:fr_row + d] += (refl * grad)[None] * (RED * 0.5 + GREY * 0.12)[:, None, None] * kk
        if L.face_w is not None:
            fw, fk = L.face_w[ly, lx], L.face_k[ly, lx]
            reg[:] = reg * (1 - 0.96 * fw[None]) + (WHITE * (0.45 + 0.55 * vis))[:, None, None] * (0.96 * fw[None])
            reg *= (1 - 0.98 * fk[None])
    return frame


def world_image(f: float) -> Image.Image:
    fr = render_world(f)
    return Image.fromarray((np.clip(fr, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8), "RGB")


# ===================================================================== build
def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: world_image(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.12, pan_start=(0.0, 0.0), pan_end=(-0.8, 0.0), ease="ease_in_out"))


def check_layout() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        for name, pose, fl, sc in all_actors(f):
            x0, y0, x1, y1 = skeleton_bounds(pose, ST.scaled(sc), fl)
            if name == "P":
                lo = [min(lo[0], x0), min(lo[1], y0)]; hi = [max(hi[0], x1), max(hi[1], y1)]
            assert y1 <= 1447, f"{name} reaches into the caption zone at f{f}"
    ok = lo[0] >= l and hi[0] <= r and lo[1] >= t
    return {"isolated_character_bounds": [round(v) for v in (*lo, *hi)], "safe": [l, t, r, 1447], "ok": ok}


def to_png(img: Image.Image, R: SceneRenderer, f: int, path: Path, zoom_pan=None):
    x = R._planar(np.asarray(img.resize((cfg.width, cfg.height), Image.LANCZOS), np.float32))
    x = R.grade.apply(x, f)
    Image.fromarray((np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    global STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=int, help="render only these world frames to qa/s02_dev_*.png")
    a = ap.parse_args()
    STAGE = Stage(cfg, ST, [LAMP_GROUP, LAMP_SOLO], floor=FLOOR)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        for f in a.stills:
            to_png(world_image(f), R, f, cfg.path(f"qa/s02_dev_{f:03d}.png"))
        return
    lay = check_layout()
    print("layout:", lay)
    if not lay["ok"]:
        sys.exit("isolated character leaves the safe area")
    scene = build_scene()
    out = cfg.path("scenes/S02.mp4")
    with FFmpegWriter(out, cfg) as w:
        n = R.render_scene(scene, w)                   # strict: raises unless exactly 150 frames
    p = qa_scene.probe(out)
    sync = qa_scene.verify_caption_sync(out, scene.captions, cfg)
    persists = qa_scene.persists_to_end(out, scene.captions[-1], cfg)
    ok = qa_scene.report(p, sync, persists) and n == 150
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf", f"select=eq(n\\,{A_EXCL + 22})",
                    "-frames:v", "1", str(cfg.path("qa/S02_preview.png"))], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf",
                    "select='eq(n\\,6)+eq(n\\,30)+eq(n\\,50)+eq(n\\,72)+eq(n\\,96)+eq(n\\,118)+eq(n\\,136)+eq(n\\,149)',scale=270:480,tile=8x1",
                    "-frames:v", "1", str(cfg.path("qa/S02_contact.png"))], check=True)
    to_png(world_image(A_BEING + 14), R, 0, cfg.path("assets/visuals/visual_02.png"))
    qa_scene.voice_preview(out, cfg.path("qa/S02_with_voice.mp4"), cfg, 2, PLAN.spill_out)
    print("S02 VERIFIED" if ok else "S02 FAILED VERIFICATION")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
