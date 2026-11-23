#!/usr/bin/env python3
"""Scene S11 - THE REVELATION.  "Sometimes, controlling your choices starts by making you feel you have none."

A free stickman stands under a spotlight with three glowing choices floating above him (arrows: left, ahead, right).
"starts by making you feel": thin red strings drop from above and attach to his head, shoulders, hands and knees. The choices
dim; he tests the strings. Pause - then "you have none.": the choices vanish and the strings pull taut, lifting him slightly and
freezing him. Timing from the approved script + real voiceover. The first caption group started in S10 (carried in);
the last group ("you have none.") carries on into S12.
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
WHITE = np.asarray(cfg.rgb01("primary_text"), np.float32)
PLAN = load_scene(11)
assert PLAN.id == "S11"
FR = {}
for i, w in enumerate(PLAN.words):
    FR.setdefault(w["word"].strip(",:.'").lower(), []).append((w["start"], w["end"]))
A_CHOICES = FR["choices"][0][0]
A_STARTS, A_MAKING, A_YOU1, A_FEEL = FR["starts"][0][0], FR["making"][0][0], FR["you"][0][0], FR["feel"][0][0]
A_YOU2, A_HAVE, A_NONE = FR["you"][1][0], FR["have"][0][0], FR["none"][0][0]
LAMPS = [Lamp(pos=(540, 40), axis=(540, 990), sigma=0.30, gain=1.6, reach=1100, halo=250),
         Lamp(pos=(560, 60), axis=(540, 800), sigma=0.60, gain=0.9, reach=1000, halo=260)]
XP = 540.0
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
LIFT_T = A_YOU2 + 6                                   # strings snap taut
CHOICES = [("←", 290.0, 480.0), ("↑", 540.0, 430.0), ("→", 790.0, 480.0)]


def taut(f: float) -> float:
    return smooth((f - LIFT_T) / 8.0)


def lift(f: float) -> float:
    return 20.0 * taut(f)


def pose_p(f: float) -> dict:
    tl = Timeline({
        "turn": [(0, 0), (150, 0)],
        "head": [(0, 0), (A_CHOICES, -6), (A_STARTS + 10, 6), (A_FEEL, -9), (A_YOU2, 0), (LIFT_T, -12), (150, -14)],
        "gaze": [(0, (0.0, 0.0)), (A_CHOICES - 6, (-0.7, -0.5)), (A_CHOICES + 12, (0.7, -0.5)), (A_STARTS + 8, (0.0, -1.0)), (A_FEEL, (0.0, -0.9)), (A_YOU2, (0.0, -0.4)), (LIFT_T, (0.0, -1.0)), (150, (0.0, -1.0))],
        "lean": [(0, 0), (A_STARTS, 0), (A_FEEL, -3), (LIFT_T, 0), (150, -2)],
        "brow": [(0, 0.0), (A_STARTS + 6, 0.4), (A_FEEL, 0.9), (A_YOU2, 0.6), (LIFT_T, 0.2), (150, 0.0)],
        "raise_": [(0, 0.5), (A_STARTS, 0.6), (A_YOU2, 0.3), (150, 0.0)],
        "wide": [(0, 0.2), (A_STARTS + 6, 0.6), (A_FEEL, 0.9), (LIFT_T, 0.8), (150, 0.7)],
        "mouth": [(0, "flat"), (A_STARTS + 6, "wavy"), (A_FEEL, "frown"), (LIFT_T, "flat"), (150, "flat")],
    })
    p = tl.at(f)
    p["hip_x"] = XP + 0.8 * math.sin(f * 0.07)
    p["lean"] += 0.6 * math.sin(f * 0.15)
    t = taut(f)
    # testing the strings: arms tug, then pulled up and left hanging
    test = smooth((f - (A_FEEL + 2)) / 10.0) * (1 - smooth((f - A_YOU2) / 6.0))
    sway = math.sin(f * 0.5)
    hR = (HANG_R[0] + 14 * math.sin(f * 0.2), HANG_R[1])
    hL = (HANG_L[0] - 14 * math.sin(f * 0.2), HANG_L[1])
    hR = (hR[0] + (60 + 14 * sway - hR[0]) * test, hR[1] + (90 - hR[1]) * test)
    hL = (hL[0] + (-60 - 14 * sway - hL[0]) * test, hL[1] + (90 - hL[1]) * test)
    hR = (hR[0] + (46 - hR[0]) * t, hR[1] + (-14 - hR[1]) * t)           # pulled up and out by the strings
    hL = (hL[0] + (-46 - hL[0]) * t, hL[1] + (-14 - hL[1]) * t)
    p["handR"], p["handL"] = hR, hL
    p["footL"], p["footR"] = (-26.0 + 8 * t, 347.0 - 10 * t), (26.0 - 8 * t, 347.0 - 10 * t)     # legs dangle
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (20, 64, 96))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR - lift(f), 1.0, {"rim": "top", "wall_shadow": False})]


def lights(f: float):
    dim = 1 - 0.30 * smooth((f - LIFT_T) / 14.0)
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21)) * dim
    return [k, k * 0.7]


def string_points(f: float):
    """Attachment points on his body (final px) for the strings."""
    pts = solve(pose_p(f), ST, FLOOR - lift(f))
    hx, hy = pts["head"]
    return [(hx, hy - ST.head_r), pts["shL"], pts["shR"], pts["haL"], pts["haR"], pts["knL"], pts["knR"]]


STRING_X = [540.0, 440.0, 640.0, 330.0, 750.0, 470.0, 610.0]


def fx(frame, f, acts, ks):
    t = taut(f)
    # ---- the three choices above him
    vis = smooth((f - (A_CHOICES - 4)) / 8.0) * (1 - smooth((f - (A_YOU2 - 10)) / 8.0 * 0 + 0))
    dim = 1.0 - 0.75 * smooth((f - A_STARTS) / 30.0)
    gone = 1.0 - smooth((f - (A_NONE - 4)) / 8.0)
    for g, x, y in CHOICES:
        a = smooth((f - (A_CHOICES - 4)) / 8.0) * dim * gone
        if a <= 0.01:
            continue
        bob = 6 * math.sin(f * 0.1 + x)
        fxlib.paint_ellipse(frame, S, x, y + bob, 58, 58, RED, alpha=0.7 * a, mode="add", outline=4, glow=6, glow_gain=0.7)
        fxlib.paint_text(frame, S, cfg, g, x, y + bob, 78, WHITE, alpha=0.95 * a, mode="add", glow=5, glow_gain=0.5)
    # ---- the strings
    P = string_points(f)
    for i, ((ax, ay), sx) in enumerate(zip(P, STRING_X)):
        t0 = A_STARTS + 3 + i * 5
        drop = smooth((f - t0) / 14.0)
        if drop <= 0:
            continue
        y_end = -40 + (ay + 40) * drop
        slack = (1 - t) * 14 * math.sin(f * 0.18 + i) * (1 if drop >= 1 else 0.3) + (1 - t) * 8
        mid = ((sx + ax) / 2 + slack, (y_end - 40) / 2 + 20)
        pts = [(sx, -40), mid, (ax * drop + sx * (1 - drop), y_end)]
        fxlib.paint_line(frame, S, pts, 3.2, RED * 1.0, alpha=0.95 * (0.7 + 0.3 * t), mode="add", glow=4, glow_gain=0.7)
        if drop >= 1.0:
            fxlib.glow_disc(frame, S, ax, ay, 9, RED, 1.0 + 1.2 * t)                  # attachment point
    if t > 0.01:                                                                       # the snap: a ring runs down the strings
        age = f - LIFT_T
        if 0 <= age < 24:
            fxlib.paint_ellipse(frame, S, 540, P[0][1] + 40, 30 + age * 24, 12 + age * 8, RED, alpha=0.9 * math.exp(-age / 8.0), mode="add", outline=3, glow=6, glow_gain=0.8)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.16, pan_start=(0.0, 0.0), pan_end=(0.0, -0.4), ease="ease_in_out"))


def layout_check() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        x0, y0, x1, y1 = skeleton_bounds(pose_p(f), ST, FLOOR - lift(f))
        lo = [min(lo[0], x0), min(lo[1], y0)]; hi = [max(hi[0], x1), max(hi[1], y1)]
    return {"bounds": [round(v) for v in (*lo, *hi)], "ok": lo[0] >= l and hi[0] <= r and lo[1] >= t and hi[1] <= 1447}


WORLD = None


def main():
    global WORLD
    ap = argparse.ArgumentParser(); ap.add_argument("--stills", nargs="*", type=int)
    a = ap.parse_args()
    stage = Stage(cfg, ST, LAMPS, floor=FLOOR, door=None)
    WORLD = common.make_world_image(stage, ST, actors, lights, fx=fx)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        common.stills(WORLD, R, a.stills, "s11"); return
    ok = common.run(11, build_scene, WORLD, R, preview_f=A_FEEL + 14, key_f=LIFT_T + 10, contact_fs=[6, 30, 56, 80, 100, 118, 134, 148], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
