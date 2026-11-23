#!/usr/bin/env python3
"""Scene S10 - PSYCHOLOGICAL REACTANCE.  "But threaten someone's freedom, and reactance may make them push back."

Red bars slam down around a free stickman ("threaten someone's freedom"). He goes from fear to anger ("reactance"), grips two bars,
bends them apart and shatters them ("push back"). The last caption group of line 11 ("Sometimes, controlling your choices")
starts in the final 13 frames and continues into S11. Timing from the approved script + real voiceover.
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
PLAN = load_scene(10)
assert PLAN.id == "S10"
FR = {w["word"].strip(",:.'").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_THREAT, A_FREEDOM, A_AND, A_REACT, A_MAY, A_PUSH, A_BACK = (FR[k][0] for k in ("threaten", "freedom", "and", "reactance", "may", "push", "back"))
LAMPS = [Lamp(pos=(540, 40), axis=(540, 990), sigma=0.30, gain=1.5, reach=1100, halo=250),
         Lamp(pos=(560, 60), axis=(540, 800), sigma=0.60, gain=1.0, reach=1000, halo=260)]
XP = 540.0
BARS = [260.0, 330.0, 400.0, 470.0, 610.0, 680.0, 750.0, 820.0]
INNER = (470.0, 610.0)
YH = 880.0                                          # height at which he grips the bars
BREAK = A_BACK + 1                                  # the bars shatter on "back."
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)


def bar_drop(i: int) -> float:
    return A_THREAT + i * 2.5


def push(f: float) -> float:
    return 150.0 * smooth((f - (A_MAY + 4)) / (BREAK - A_MAY - 4)) if f < BREAK else 150.0


def bend_x(i: int, y: float, f: float) -> float:
    x = BARS[i]
    s = -1.0 if x < XP else 1.0
    gain = 1.0 if x in INNER else 0.55 if abs(x - XP) < 150 else 0.3
    return x + s * push(f) * gain * math.exp(-((y - YH) / 230.0) ** 2)


def pose_p(f: float) -> dict:
    anger = smooth((f - A_AND) / 24.0)
    tl = Timeline({
        "turn": [(0, 0), (150, 0)],
        "head": [(0, 0), (A_THREAT, 0), (A_THREAT + 4, -8), (A_FREEDOM + 6, 2), (A_REACT, 0), (A_PUSH, 10), (BREAK, -6), (150, 4)],
        "gaze": [(0, (0.0, 0.0)), (A_THREAT + 8, (-0.8, 0.0)), (A_FREEDOM, (0.8, 0.0)), (A_AND, (0.0, 0.0)), (150, (0.0, 0.0))],
        "lean": [(0, 0), (A_THREAT + 2, -6), (A_FREEDOM + 6, -2), (A_PUSH - 6, 5), (BREAK, 10), (BREAK + 14, 2), (150, 3)],
        "brow": [(0, 0.0), (A_THREAT + 3, 1.0), (A_FREEDOM + 8, 0.6), (A_AND + 6, -0.4), (A_REACT + 10, -1.0), (150, -0.6)],
        "raise_": [(0, 0.2), (A_THREAT + 3, 0.9), (A_FREEDOM + 8, 0.4), (A_AND, 0.0), (150, 0.0)],
        "wide": [(0, 0.0), (A_THREAT + 3, 1.0), (A_FREEDOM + 10, 0.4), (A_AND, 0.0), (BREAK, 0.8), (150, 0.3)],
        "mouth": [(0, "smile"), (A_THREAT + 2, "o"), (A_FREEDOM + 8, "wavy"), (A_AND + 8, "frown"), (BREAK - 4, "o"), (BREAK + 16, "frown"), (150, "frown")],
        "shrug": [(0, 0), (A_THREAT + 3, 0.8), (A_FREEDOM + 10, 0.3), (A_AND, 0), (150, 0)],
    })
    p = tl.at(f)
    p["hip_x"] = XP + 1.0 * math.sin(f * 0.07) + (2.2 * math.sin(f * 2.1) * anger * (f < BREAK))
    p["lean"] += 0.6 * math.sin(f * 0.15)
    # hands: free swing -> up in fear -> grip the bars -> push them apart -> drop
    free = (HANG_R[0] + 18 * math.sin(f * 0.2), HANG_R[1]), (HANG_L[0] - 18 * math.sin(f * 0.2), HANG_L[1])
    fear = smooth((f - A_THREAT) / 4.0) * (1 - smooth((f - (A_AND - 4)) / 10.0))
    grip = smooth((f - (A_AND + 4)) / 14.0) * (1 - smooth((f - (BREAK + 10)) / 12.0))
    tR = common.hand_target(p, "R", (bend_x(BARS.index(INNER[1]), YH, f), YH), ST, FLOOR)
    tL = common.hand_target(p, "L", (bend_x(BARS.index(INNER[0]), YH, f), YH), ST, FLOOR)
    up_r, up_l = (62.0, -20.0), (-62.0, -20.0)
    hR = (free[0][0] * (1 - fear) + up_r[0] * fear, free[0][1] * (1 - fear) + up_r[1] * fear)
    hL = (free[1][0] * (1 - fear) + up_l[0] * fear, free[1][1] * (1 - fear) + up_l[1] * fear)
    p["handR"] = (hR[0] + (tR[0] - hR[0]) * grip, hR[1] + (tR[1] - hR[1]) * grip)
    p["handL"] = (hL[0] + (tL[0] - hL[0]) * grip, hL[1] + (tL[1] - hL[1]) * grip)
    # braced legs while pushing
    brace = smooth((f - A_MAY) / 10.0) * (1 - smooth((f - (BREAK + 14)) / 10.0))
    p["footL"], p["footR"] = (-26.0 - 22 * brace, 347.0 - 12 * brace), (26.0 + 22 * brace, 347.0 - 12 * brace)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (18, 52))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR, 1.0, {"rim": "top", "wall_shadow": False})]


def lights(f: float):
    flare = 1.0 + 0.9 * math.exp(-((f - BREAK) / 5.0) ** 2) * (f >= BREAK - 2)
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21)) * flare
    return [k, k * 0.7]


def bar_polyline(i: int, f: float, y_top: float, y_bot: float):
    ys = np.arange(y_top, y_bot + 1, 50.0)
    return [(bend_x(i, float(y), f), float(y)) for y in ys] + [(bend_x(i, y_bot, f), y_bot)]


def draw_bar(frame, pts, a, thick=13.0):
    fxlib.paint_line(frame, S, pts, thick, (0.035, 0.03, 0.04), alpha=0.97 * a)
    fxlib.paint_line(frame, S, pts, thick * 0.38, RED * 1.05, alpha=0.9 * a, mode="add", glow=5, glow_gain=0.9)


def fx(frame, f, acts, ks):
    anger = smooth((f - A_AND) / 24.0)
    pts = solve(acts[0][1], ST, FLOOR)
    cx, cy = (pts["hip"][0] + pts["spine"][0]) / 2, (pts["hip"][1] + pts["spine"][1]) / 2
    if anger > 0:
        fxlib.glow_disc(frame, S, cx, cy, 170, RED, 0.42 * anger * (0.8 + 0.2 * math.sin(f * 0.8)) * (1 if f < BREAK + 20 else 1 - smooth((f - BREAK - 20) / 14.0)))
    for i, x in enumerate(BARS):
        t0 = bar_drop(i)
        if f < t0:
            continue
        u = min(1.0, (f - t0) / 5.0)
        tip = -60 + (FLOOR - -60) * (u * u)
        if u >= 1.0 and 0 <= f - t0 - 5 < 14:
            age = f - t0 - 5
            fxlib.paint_ellipse(frame, S, x, FLOOR + 6, 14 + age * 5, 5 + age * 1.6, RED, alpha=0.9 * math.exp(-age / 5.0), mode="add", outline=3, glow=4, glow_gain=0.7)
        if f < BREAK:
            draw_bar(frame, bar_polyline(i, f, -60, tip), 1.0)
            continue
        age = f - BREAK
        inner = x in INNER
        if inner:   # inner bars snap at the grip point: halves fly apart and fade
            gap = 22 + age * 11.0
            fade = max(0.0, 1 - age / 26.0)
            draw_bar(frame, bar_polyline(i, BREAK, -60, YH - gap), fade)
            draw_bar(frame, bar_polyline(i, BREAK, YH + gap, FLOOR), fade)
            for j in range(9):                                        # shards
                ang = (j / 9.0) * 2 * math.pi + i
                vx, vy = math.cos(ang) * 16, math.sin(ang) * 14 - 6
                fxlib.glow_disc(frame, S, bend_x(i, YH, BREAK) + vx * age, YH + vy * age + 0.5 * 1.1 * age * age, 7, RED, 1.3 * math.exp(-age / 9.0))
        else:       # the rest of the cage dissolves
            fade = max(0.0, 1 - age / 14.0)
            if fade > 0:
                draw_bar(frame, bar_polyline(i, BREAK, -60, FLOOR), fade)
    if f >= BREAK - 1:
        age = f - BREAK + 1
        if age < 22:
            fxlib.paint_ellipse(frame, S, cx, cy, 40 + age * 36, 40 + age * 36, RED, alpha=0.9 * math.exp(-age / 8.0), mode="add", outline=5, glow=8, glow_gain=1.0)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.14, ease="ease_in_out"))


def layout_check() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        x0, y0, x1, y1 = skeleton_bounds(pose_p(f), ST, FLOOR)
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
        common.stills(WORLD, R, a.stills, "s10"); return
    ok = common.run(10, build_scene, WORLD, R, preview_f=A_REACT + 8, key_f=BREAK + 4, contact_fs=[6, 24, 44, 70, 96, 112, 120, 146], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
