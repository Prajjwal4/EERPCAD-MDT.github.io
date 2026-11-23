#!/usr/bin/env python3
"""Scene S04 - MIRRORING.  "Next, mirroring: people often unconsciously copy gestures and expressions."

Two stickmen face each other across a faint pane of glass. The one on the left makes gestures (hand to chin, scratching his
temple, a wave); the one on the right copies each of them, mirrored, 9 frames later - blank-faced, unaware. A red ripple
crosses the pane at each copy. Timing from the approved script + real voiceover (script_plan.py); the last caption group
("copy gestures and expressions.") continues into S05.
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
PLAN = load_scene(4)
assert PLAN.id == "S04"
FR = {w["word"].strip(",:.").lower(): (w["start"], w["end"]) for w in PLAN.words}
DELAY = 9                                  # frames B lags behind A
PANE_X = 540.0
LAMPS = [Lamp(pos=(250, 60), axis=(330, 990), sigma=0.32, gain=1.25, reach=1000, halo=240),
         Lamp(pos=(830, 60), axis=(750, 990), sigma=0.32, gain=1.25, reach=1000, halo=240)]

HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
CHIN, TEMPLE_HI, TEMPLE_LO, WAVE_HI, WAVE_LO = (12.0, -60.0), (36.0, -90.0), (34.0, -78.0), (72.0, -132.0), (82.0, -118.0)
A_MIR, A_PEOPLE, A_OFTEN, A_UNC, A_COPY, A_GEST, A_AND = (FR[k][0] for k in ("mirroring", "people", "often", "unconsciously", "copy", "gestures", "and"))

A_TL = Timeline({
    "head": [(0, 6), (A_MIR + 2, 6), (A_MIR + 10, -10), (A_MIR + 32, -10), (A_MIR + 40, 8), (A_UNC + 10, 8), (A_COPY + 2, -6), (150, 0)],
    "handR": [(0, HANG_R), (A_MIR + 2, HANG_R), (A_MIR + 12, CHIN), (A_MIR + 32, CHIN), (A_MIR + 40, HANG_R), (A_OFTEN + 6, HANG_R),
              (A_UNC + 6, TEMPLE_HI), (A_UNC + 10, TEMPLE_LO), (A_UNC + 14, TEMPLE_HI), (A_UNC + 18, TEMPLE_LO), (A_UNC + 24, HANG_R),
              (A_COPY + 2, HANG_R), (A_COPY + 10, WAVE_HI), (A_COPY + 16, WAVE_LO), (A_COPY + 22, WAVE_HI), (A_COPY + 28, WAVE_LO), (150, WAVE_HI)],
    "handL": [(0, HANG_L), (150, HANG_L)],
    "lean": [(0, 1), (A_MIR, 1), (A_MIR + 14, -2), (A_UNC, 2), (A_COPY + 10, -2), (150, 0)],
    "gaze": [(0, (1.0, 0.0)), (150, (1.0, 0.0))],
    "brow": [(0, 0.0), (A_MIR + 10, 0.4), (A_MIR + 32, 0.4), (A_COPY, 0.0), (150, 0.0)],
    "raise_": [(0, 0.0), (A_MIR + 10, 0.5), (A_MIR + 32, 0.5), (A_AND, 0.0), (A_AND + 10, 0.6), (150, 0.6)],
    "mouth": [(0, "flat"), (A_MIR + 6, "o"), (A_MIR + 40, "flat"), (A_COPY, "smile"), (150, "smile")],
    "wide": [(0, 0.0), (150, 0.0)],
})


def pose_a(f: float) -> dict:
    p = A_TL.at(f)
    p.update(hip_x=330.0 + 1.2 * math.sin(f * 0.07), turn=62.0)
    p["lean"] += 0.7 * math.sin(f * 0.15)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (16, 71, 128))
    return p


def mirror(p: dict, x: float) -> dict:
    q = dict(p)
    q["handL"], q["handR"] = (-p["handR"][0], p["handR"][1]), (-p["handL"][0], p["handL"][1])
    q["turn"], q["head"], q["lean"] = -p["turn"], -p["head"], -p["lean"]
    q["gaze"] = (-p["gaze"][0], p["gaze"][1])
    q["hip_x"] = x + 1.2 * math.sin(f_sin * 0.07 + 1.0)
    return q


f_sin = 0.0


def pose_b(f: float) -> dict:
    global f_sin
    f_sin = f
    q = mirror(pose_a(f - DELAY), 750.0)
    q["mouth"] = "flat" if f < A_AND + DELAY + 2 else "smile"                 # unaware: blank face until the copy of the smile
    q["brow"], q["raise_"] = 0.0, (0.0 if f < A_AND + DELAY else 0.5)
    q["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (30, 88, 140))
    return q


def actors(f: float):
    return [("P", pose_a(f), FLOOR, 1.0, {"rim": "left", "wall_shadow": False}),
            ("B", pose_b(f), FLOOR, 1.0, {"rim": "right", "wall_shadow": False, "vis": 0.9})]


def lights(f: float):
    k = smooth(f / 10.0) * (0.94 + 0.04 * math.sin(f * 0.21) + 0.02 * math.sin(f * 0.57 + 1.0))
    return [k, k * (0.97 + 0.03 * math.sin(f * 0.37 + 2.0))]


COPY_TIMES = [A_MIR + 12 + DELAY, A_UNC + 6 + DELAY, A_COPY + 10 + DELAY]       # B's hand arrives: ripple on the pane


def pre_fx(frame, f, ks):
    """The pane of glass between them (behind the characters)."""
    k = min(ks)
    fxlib.paint_rect(frame, S, PANE_X - 3, 360, PANE_X + 3, FLOOR + 6, RED, alpha=0.20 * k, mode="add", glow=10, glow_gain=0.6)
    fxlib.paint_rect(frame, S, PANE_X - 1.2, 360, PANE_X + 1.2, FLOOR + 6, WHITE, alpha=0.38 * k, mode="add")
    sweep = (f * 5.0) % 1500 - 200                                           # a slow sheen sliding down the glass
    fxlib.paint_poly(frame, S, [(PANE_X - 30, 360 + sweep), (PANE_X + 30, 360 + sweep - 60), (PANE_X + 30, 360 + sweep + 60), (PANE_X - 30, 360 + sweep + 120)],
                     WHITE, alpha=0.05 * k, mode="add")


def fx(frame, f, acts, ks):
    P = {n: (pose, fl, sc) for n, pose, fl, sc, *_ in acts}
    pa, pb = solve(P["P"][0], ST, FLOOR), solve(P["B"][0], ST, FLOOR)
    # red ripples on the pane when the copy lands, and a thin echo line between the two raised hands
    for t in COPY_TIMES:
        age = f - t
        if 0 <= age < 26:
            hy = pb["haL"][1]
            fxlib.paint_ellipse(frame, S, PANE_X, hy, 8 + age * 2.2, 22 + age * 6.0, RED, alpha=0.9 * math.exp(-age / 9.0), mode="add",
                                outline=3.0, glow=6, glow_gain=0.8)
    raised = max(0.0, min(1.0, (FLOOR - 340 - min(pa["haR"][1], pb["haL"][1])) / 380.0))
    if raised > 0.2:
        ax, ay, bx, by = pa["haR"][0], pa["haR"][1], pb["haL"][0], pb["haL"][1]
        fxlib.paint_line(frame, S, [(ax, ay), ((ax + bx) / 2, min(ay, by) - 40), (bx, by)], 2.5, RED, alpha=0.55 * raised, mode="add", glow=5, glow_gain=0.7)


STAGE = None


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.10, pan_start=(0.0, 0.0), pan_end=(0.0, 0.0), ease="ease_in_out"))


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
    global STAGE, WORLD
    ap = argparse.ArgumentParser(); ap.add_argument("--stills", nargs="*", type=int)
    a = ap.parse_args()
    STAGE = Stage(cfg, ST, LAMPS, floor=FLOOR, door=None)
    WORLD = common.make_world_image(STAGE, ST, actors, lights, pre_fx=pre_fx, fx=fx)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        common.stills(WORLD, R, a.stills, "s04"); return
    ok = common.run(4, build_scene, WORLD, R, preview_f=A_UNC + 22, key_f=A_UNC + 24, contact_fs=[6, 34, 54, 80, 106, 126, 142, 149], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
