#!/usr/bin/env python3
"""Scene S09 - SMALL AGREEMENTS.  "Agreeing once can make a larger request easier to accept."

A second stickman hands over boxes that grow: a small one ("Agreeing once"), a medium one ("a larger request"), a big one
("easier to accept"). The first stickman nods yes every time - a red check pops up - and each box is easier to accept than the
last; the pile in his arms grows while his smile does too. Timing from the approved script + real voiceover.
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
PLAN = load_scene(9)
assert PLAN.id == "S09"
FR = {w["word"].strip(",:.'").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_AGREE, A_ONCE, A_CAN, A_LARGER, A_REQUEST, A_EASIER, A_TO, A_ACCEPT = (FR[k][0] for k in ("agreeing", "once", "can", "larger", "request", "easier", "to", "accept"))
LAMPS = [Lamp(pos=(300, 40), axis=(380, 990), sigma=0.34, gain=1.3, reach=1000, halo=240),
         Lamp(pos=(800, 40), axis=(720, 990), sigma=0.34, gain=1.3, reach=1000, halo=240)]
XP, XG = 330.0, 760.0
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
# (offer starts, arrives, box w, box h)
BOXES = [(A_AGREE + 4, A_ONCE - 2, 70.0, 36.0), (A_CAN + 8, A_LARGER + 12, 104.0, 66.0), (A_EASIER + 6, A_ACCEPT - 2, 142.0, 100.0)]
STACK_X, STACK_BOTTOM = XP + 92.0, 990.0


def stack_geometry(upto: int):
    """Boxes resting in his arms: smallest at the bottom, each new one on top. Returns [(cx, y0, y1, w)] for the first `upto`."""
    out, y = [], STACK_BOTTOM
    for i in range(upto):
        _, _, w, h = BOXES[i]
        out.append((STACK_X, y - h, y, w))
        y -= h
    return out


def accepted(f: float) -> int:
    return sum(1 for b in BOXES if f >= b[1] + 3)


def box_pos(i: int, f: float):
    """Box i in flight: from the asker's hands to P's stack slot."""
    t0, t1, w, h = BOXES[i]
    if f < t0 or f >= t1 + 3:
        return None
    u = smooth((f - (t0 + 6)) / max(t1 - t0 - 6, 1))
    slot = stack_geometry(i + 1)[-1]
    sx, sy = slot[0], (slot[1] + slot[2]) / 2
    ax, ay = XG - 80, 930.0
    return ax + (sx - ax) * u, ay + (sy - ay) * u - 24 * math.sin(math.pi * u), w, h


def pose_p(f: float) -> dict:
    nods = [(b[1], 12.0 + 4 * i) for i, b in enumerate(BOXES)]
    head = 0.0
    for t, a in nods:
        head += a * math.exp(-((f - t) / 3.5) ** 2)
    n = accepted(f)
    p = {"turn": 62.0, "hip_x": XP + 1.0 * math.sin(f * 0.07), "head": head, "gaze": (1.0, 0.0),
         "brow": 0.0, "raise_": 0.2 + 0.2 * n, "wide": 0.0,
         "mouth": ["flat", "smile", "smile", "smile"][n] if n < 3 else "smile", "lean": -2.0 * n + 0.6 * math.sin(f * 0.15)}
    # hands: relaxed, then holding the stack from below/side
    hold = smooth((f - (BOXES[0][1] - 6)) / 8.0)
    slots = stack_geometry(max(1, n))
    top_y = min(s[1] for s in slots)
    target_r = (STACK_X + BOXES[max(0, n - 1)][2] / 2 - 4, (top_y + STACK_BOTTOM) / 2 + 6)
    target_l = (STACK_X - BOXES[max(0, n - 1)][2] / 2 + 6, (top_y + STACK_BOTTOM) / 2 + 22)
    tr = common.hand_target(p, "R", target_r, ST, FLOOR)
    tl = common.hand_target(p, "L", target_l, ST, FLOOR)
    # while a box is in flight, reach out to receive it
    for i, b in enumerate(BOXES):
        if b[0] + 4 <= f < b[1] + 3:
            r = smooth((f - (b[0] + 4)) / 6.0)
            bp = box_pos(i, f)
            if bp:
                tr = common.hand_target(p, "R", (bp[0] + 18, bp[1] + 4), ST, FLOOR)
                tl = common.hand_target(p, "L", (bp[0] - 18, bp[1] + 14), ST, FLOOR)
    p["handR"] = (HANG_R[0] + (tr[0] - HANG_R[0]) * hold, HANG_R[1] + (tr[1] - HANG_R[1]) * hold)
    p["handL"] = (HANG_L[0] + (tl[0] - HANG_L[0]) * hold, HANG_L[1] + (tl[1] - HANG_L[1]) * hold)
    sag = 7.0 * n                                         # the load makes his knees bend a little more each time
    p["footL"], p["footR"] = (-26.0, 347.0 - sag), (26.0, 347.0 - sag)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (30, 76, 122))
    return p


def pose_g(f: float) -> dict:
    p = {"turn": -62.0, "hip_x": XG + 1.0 * math.sin(f * 0.07 + 1.0), "gaze": (-1.0, 0.0), "head": 3 * math.sin(f * 0.2), "lean": 1.0,
         "mouth": "smile", "brow": 0.0, "raise_": 0.0, "blink": max(max(0.0, 1 - abs(f - c) / 2.0) for c in (20, 64, 110))}
    hand_h, hand_l = HANG_R, HANG_L
    tl, tr = hand_l, hand_h
    for i, b in enumerate(BOXES):
        if b[0] - 6 <= f < b[1] + 6:
            r = smooth((f - (b[0] - 6)) / 6.0) * (1 - smooth((f - (b[1] - 4)) / 8.0))
            bp = box_pos(i, f) or (XG - 80, 930.0, 0, 0)
            a = common.hand_target(p, "L", (bp[0] + 20, bp[1] + 4), ST, FLOOR)
            c = common.hand_target(p, "R", (bp[0] + 16, bp[1] + 16), ST, FLOOR)
            tl = (tl[0] + (a[0] - tl[0]) * r, tl[1] + (a[1] - tl[1]) * r)
            tr = (tr[0] + (c[0] - tr[0]) * r, tr[1] + (c[1] - tr[1]) * r)
    p["handL"], p["handR"] = tl, tr
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR, 1.0, {"rim": "left", "wall_shadow": False}),
            ("G", pose_g(f), FLOOR, 1.0, {"rim": "right", "wall_shadow": False, "vis": 0.9})]


def lights(f: float):
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21))
    return [k, k * (0.97 + 0.03 * math.sin(f * 0.37 + 2.0))]


def draw_box(frame, cx, cy, w, h, a=1.0):
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    fxlib.paint_rect(frame, S, x0, y0, x1, y1, (0.11, 0.06, 0.07), alpha=0.97 * a, radius=4)
    fxlib.paint_rect(frame, S, x0, y0, x1, y1, RED, alpha=0.85 * a, mode="add", radius=4, outline=3, glow=4, glow_gain=0.7)
    fxlib.paint_line(frame, S, [(cx, y0), (cx, y1)], max(5.0, w * 0.10), RED * 0.7, alpha=0.55 * a, mode="add")          # tape


def fx(frame, f, acts, ks):
    P = solve(acts[0][1], ST, FLOOR)
    n = accepted(f)
    for i, (cx, y0, y1, w) in enumerate(stack_geometry(n)):
        draw_box(frame, cx, (y0 + y1) / 2, w, y1 - y0 - 3)
    for i in range(3):
        bp = box_pos(i, f)
        if bp:
            draw_box(frame, bp[0], bp[1], bp[2], bp[3])
    # the red tick that pops up each time he agrees
    for i, b in enumerate(BOXES):
        age = f - (b[1] + 2)
        if 0 <= age < 26:
            hx, hy = P["head"]
            px = 40 + 18 * i
            fxlib.paint_text(frame, S, cfg, "✓", hx + 4, hy - 108 - age * 1.6, px * 1.8, RED * 1.15, alpha=math.sin(math.pi * min(1.0, age / 26.0)) ** 0.6, mode="add", glow=6, glow_gain=1.0)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.12, pan_start=(0.0, 0.0), pan_end=(-0.4, 0.0), ease="ease_in_out"))


def layout_check() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        for fn in (pose_p, pose_g):
            x0, y0, x1, y1 = skeleton_bounds(fn(f), ST, FLOOR)
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
        common.stills(WORLD, R, a.stills, "s09"); return
    ok = common.run(9, build_scene, WORLD, R, preview_f=A_LARGER + 22, key_f=A_ACCEPT + 6, contact_fs=[6, 22, 44, 62, 84, 100, 116, 140], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
