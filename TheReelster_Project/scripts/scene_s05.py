#!/usr/bin/env python3
"""Scene S05 - THE CHAMELEON EFFECT.  "This Chameleon Effect can help interactions feel more natural."

Continues S04: the pair are still copying a wave. A chameleon appears on a branch above them and its skin shifts from red to
camouflage. As "interactions feel more natural" lands, the second character's stiffness melts into easy synchrony with the
first, his cold rim light warms to the same red, a thread of warm light links them. The first 38 frames still carry
"copy gestures and expressions." from S04.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fxlib
import scene_common as common
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, Scene, SceneRenderer
from script_plan import load_scene
from stage import Lamp, Stage
from stickman import Style, Timeline, skeleton_bounds, smooth, solve

cfg = load_config()
ST = Style.from_cfg(cfg)
S = ST.world_scale
FLOOR = 1340.0
RED = np.asarray(cfg.rgb01("accent_red"), np.float32)
BLUE = np.asarray(cfg.rgb01("research_blue"), np.float32)
WHITE = np.asarray(cfg.rgb01("primary_text"), np.float32)
PLAN = load_scene(5)
assert PLAN.id == "S05"
FR = {w["word"].strip(",:.").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_THIS, A_CHAM, A_EFFECT, A_CAN, A_INTER, A_FEEL, A_NAT = (FR[k][0] for k in ("this", "chameleon", "effect", "can", "interactions", "feel", "natural"))
LAMPS = [Lamp(pos=(250, 60), axis=(350, 990), sigma=0.32, gain=1.25, reach=1000, halo=240),
         Lamp(pos=(830, 60), axis=(730, 990), sigma=0.32, gain=1.25, reach=1000, halo=240)]
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
WAVE_HI, WAVE_LO = (56.0, -126.0), (64.0, -112.0)
XA, XB = 350.0, 730.0


def blend(p1: dict, p2: dict, w: float) -> dict:
    out = {}
    for k, v in p1.items():
        v2 = p2.get(k, v)
        if isinstance(v, str):
            out[k] = v2 if w > 0.5 else v
        elif isinstance(v, tuple):
            out[k] = tuple(a + (b - a) * w for a, b in zip(v, v2))
        else:
            out[k] = v + (v2 - v) * w
    return out


def mirror(p: dict, x: float) -> dict:
    q = dict(p)
    q["handL"], q["handR"] = (-p["handR"][0], p["handR"][1]), (-p["handL"][0], p["handL"][1])
    q["turn"], q["head"], q["lean"] = -p["turn"], -p["head"], -p["lean"]
    q["gaze"] = (-p["gaze"][0], p["gaze"][1])
    q["hip_x"] = x
    return q


def chat(f: float, ph: float, amp: float) -> dict:
    return {"handR": (34 + amp * (10 + 8 * math.sin(0.33 * f + ph)), 200 - amp * 34 * max(0.0, math.sin(0.21 * f + ph * 1.7))),
            "handL": (-34 - amp * (4 + 6 * math.sin(0.27 * f + ph)), 214 - amp * 14 * max(0.0, math.sin(0.19 * f + ph))),
            "head": 5 * math.sin(0.2 * f + ph) * amp, "lean": 1.5 * math.sin(0.17 * f + ph)}


def pose_a(f: float) -> dict:
    wv = 1 - smooth((f - 24) / 12.0)                                   # still waving at the start (continuity with S04)
    c = chat(f, 0.0, 1.0)
    wave = (54 + 6 * math.sin(0.7 * f), -122 + 8 * math.cos(0.7 * f))
    p = {"turn": 34.0, "hip_x": XA + 1.2 * math.sin(f * 0.07), "gaze": (1.0, 0.0), "mouth": "o" if math.sin(0.8 * f) > 0.35 else "smile",
         "brow": 0.0, "raise_": 0.5, "handL": c["handL"], "head": c["head"] * smooth((f - 30) / 10.0), "lean": c["lean"]}
    p["handR"] = (c["handR"][0] + (wave[0] - c["handR"][0]) * wv, c["handR"][1] + (wave[1] - c["handR"][1]) * wv)
    p["blink"] = max(max(0.0, 1 - abs(f - c_) / 2.0) for c_ in (20, 74, 128))
    return p


def sync_w(f: float) -> float:
    return smooth((f - (A_CAN - 2)) / 44.0)


def pose_b(f: float) -> dict:
    # copies A's wave 9 frames late at first, then goes stiff and unnatural, then melts into synchrony with A
    a_lag = pose_a(f - 9)
    copy = mirror(a_lag, XB)
    stiff = {"turn": -50.0, "hip_x": XB, "handR": HANG_R, "handL": HANG_L, "head": 0.0, "lean": 0.0, "gaze": (-1.0, 0.0), "mouth": "flat",
             "brow": 0.0, "raise_": 0.0}
    own = blend(copy, stiff, smooth((f - 26) / 14.0))
    nat = mirror(pose_a(f - 5), XB)
    nat["mouth"], nat["raise_"] = ("smile" if math.sin(0.8 * f + 1.5) > -0.2 else "o"), 0.5
    p = blend(own, nat, sync_w(f))
    p["hip_x"] = XB + 1.2 * math.sin(f * 0.07 + 1.0)
    p["blink"] = max(max(0.0, 1 - abs(f - c_) / 2.0) for c_ in (34, 92, 134))
    return p


def actors(f: float):
    w = sync_w(f)
    cold = 0.55 * BLUE + 0.30 * RED
    return [("P", pose_a(f), FLOOR, 1.0, {"rim": "left", "wall_shadow": False, "rim_color": RED}),
            ("B", pose_b(f), FLOOR, 1.0, {"rim": "right", "wall_shadow": False, "vis": 0.6 + 0.4 * w, "rim_color": cold + (RED - cold) * w})]


def lights(f: float):
    k = smooth(f / 8.0) * (0.80 + 0.30 * sync_w(f) + 0.03 * math.sin(f * 0.21))
    return [k, k * (0.97 + 0.03 * math.sin(f * 0.37 + 2.0))]


def skin(f: float) -> np.ndarray:
    """Chameleon colour: red -> camouflage grey-blue -> red again as the interaction becomes 'natural'."""
    camo = np.asarray([0.16, 0.17, 0.22], np.float32)
    t1 = smooth((f - A_CHAM) / 12.0) - smooth((f - A_EFFECT - 14) / 10.0) * 0.0
    to_camo = smooth((f - (A_EFFECT - 4)) / 14.0)
    back = smooth((f - (A_INTER + 4)) / 22.0)
    c = RED * 0.95
    c = c + (camo - c) * to_camo * (1 - back)
    return c


def pre_fx(frame, f, ks):
    """Branch + chameleon (fades in on 'This' and keeps shifting colour)."""
    a = smooth((f - (A_THIS - 2)) / 10.0)
    if a <= 0:
        return
    col = skin(f)
    bx0, bx1, by = 330, 760, 492
    cx, cy, K = 540.0, 430.0, 1.4
    bob = 1.5 * math.sin(f * 0.12)
    X = lambda x: cx + (x - cx) * K
    Y = lambda y: cy + (y - cy) * K
    pt = lambda x, y: (X(x), Y(y))
    fxlib.paint_line(frame, S, [pt(bx0, by + 8), pt(545, by), pt(bx1, by + 14)], 11 * K, (0.10, 0.09, 0.10), alpha=0.95 * a)
    tail = [(cx - 96, cy + 8)]
    for i in range(1, 14):
        ang = math.pi * 0.9 + i * 0.52
        r = 40 - i * 2.6
        tail.append((cx - 140 + r * math.cos(ang) * 0.9, cy + 36 + r * math.sin(ang) * 0.9))
    fxlib.paint_line(frame, S, [pt(*q) for q in tail], 13 * K, col, alpha=0.95 * a)
    fxlib.paint_ellipse(frame, S, cx, cy + bob * K, 98 * K, 52 * K, col, alpha=0.95 * a)
    for dx in (-52, -14, 36, 70):                                       # skin pattern: darker patches
        fxlib.paint_ellipse(frame, S, X(cx + dx), Y(cy - 8 + bob), 15 * K, 11 * K, col * 0.55, alpha=0.8 * a)
    fxlib.paint_poly(frame, S, [pt(cx + 82, cy - 34 + bob), pt(cx + 152, cy - 8 + bob), pt(cx + 146, cy + 26 + bob), pt(cx + 78, cy + 30 + bob)], col, alpha=0.95 * a)   # head
    fxlib.paint_poly(frame, S, [pt(cx + 92, cy - 40 + bob), pt(cx + 118, cy - 66 + bob), pt(cx + 132, cy - 30 + bob)], col * 0.8, alpha=0.95 * a)                      # crest
    for lx in (-58, -22, 44, 78):
        fxlib.paint_line(frame, S, [pt(cx + lx, cy + 42 + bob), pt(cx + lx + 6, by - 6)], 9 * K, col * 0.85, alpha=0.95 * a)
    glance = math.sin(f * 0.09) * 5
    fxlib.paint_ellipse(frame, S, X(cx + 128), Y(cy - 12 + bob), 17 * K, 17 * K, col * 0.7, alpha=0.95 * a)
    fxlib.paint_ellipse(frame, S, X(cx + 128), Y(cy - 12 + bob), 17 * K, 17 * K, WHITE * 0.85, alpha=0.9 * a, outline=3)
    fxlib.paint_ellipse(frame, S, X(cx + 130 + glance), Y(cy - 12 + bob), 6 * K, 6 * K, (0.02, 0.02, 0.03), alpha=a)
    fxlib.glow_disc(frame, S, cx, cy, 150, col, 0.10 * a)


def fx(frame, f, acts, ks):
    w = sync_w(f)
    if w > 0.05:
        pa, pb = solve(acts[0][1], ST, FLOOR), solve(acts[1][1], ST, FLOOR)
        (ax, ay), (bx, by) = pa["head"], pb["head"]
        fxlib.paint_line(frame, S, [(ax + 70, ay - 30), ((ax + bx) / 2, min(ay, by) - 92), (bx - 70, by - 30)], 3.0, RED, alpha=0.65 * w, mode="add", glow=6, glow_gain=0.8)
        fxlib.glow_disc(frame, S, (ax + bx) / 2, (ay + by) / 2 + 120, 130, RED, 0.18 * w)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.10, ease="ease_in_out"))


def layout_check() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        for fn in (pose_a, pose_b):
            x0, y0, x1, y1 = skeleton_bounds(fn(f), ST, FLOOR)
            lo = [min(lo[0], x0), min(lo[1], y0)]; hi = [max(hi[0], x1), max(hi[1], y1)]
    return {"bounds": [round(v) for v in (*lo, *hi)], "ok": lo[0] >= l and hi[0] <= r and lo[1] >= t and hi[1] <= 1447}


WORLD = None


def main():
    global WORLD
    ap = argparse.ArgumentParser(); ap.add_argument("--stills", nargs="*", type=int)
    a = ap.parse_args()
    stage = Stage(cfg, ST, LAMPS, floor=FLOOR, door=None)
    WORLD = common.make_world_image(stage, ST, actors, lights, pre_fx=pre_fx, fx=fx)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        common.stills(WORLD, R, a.stills, "s05"); return
    ok = common.run(5, build_scene, WORLD, R, preview_f=A_EFFECT + 8, key_f=A_INTER + 10, contact_fs=[6, 40, 60, 80, 100, 118, 136, 148], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
