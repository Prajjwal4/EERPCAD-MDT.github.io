#!/usr/bin/env python3
"""Scene S06 - SCARCITY.  "Then there's scarcity: limited availability can make things seem more desirable."

A shelf of five glowing orbs; as "scarcity" is spoken they vanish one by one until a single orb is left in a narrow spotlight.
A stickman, indifferent at first, grows wide-eyed as it becomes the last one: he steps forward and reaches for it while the
orb pulses faster, brighter, and floats up toward him ("more desirable"). Timing from the approved script + real voiceover.
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
PLAN = load_scene(6)
assert PLAN.id == "S06"
FR = {w["word"].strip(",:.'").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_SCARCITY, A_LIMITED, A_AVAIL, A_CAN, A_MAKE, A_SEEM, A_MORE, A_DESIR = (FR[k][0] for k in ("scarcity", "limited", "availability", "can", "make", "seem", "more", "desirable"))
LAMPS = [Lamp(pos=(600, 40), axis=(720, 900), sigma=0.55, gain=1.25, reach=1000, halo=250),       # wide light on the whole shelf
         Lamp(pos=(720, 30), axis=(720, 860), sigma=0.13, gain=2.6, reach=1300, halo=190)]       # narrow spotlight on the last orb
SHELF_Y, ORB_R = 830.0, 40.0
ORBS = [(520.0, "gone"), (920.0, "gone"), (620.0, "gone"), (820.0, "gone"), (720.0, "last")]
VANISH = [A_SCARCITY + 2, A_SCARCITY + 12, A_SCARCITY + 22, A_SCARCITY + 32]                  # first four orbs disappear in this order
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)

P_TL = Timeline({
    "hip_x": [(0, 235), (A_LIMITED, 235), (A_CAN + 4, 255), (A_MAKE + 14, 392), (A_DESIR, 440), (150, 452)],
    "turn": [(0, 50), (A_CAN, 50), (A_MORE, 72), (150, 76)],
    "head": [(0, 0), (A_SCARCITY, -6), (A_LIMITED, 4), (A_MAKE, -8), (150, -10)],
    "gaze": [(0, (0.6, 0.0)), (A_SCARCITY, (1.0, -0.2)), (A_MAKE, (1.0, -0.3)), (150, (1.0, -0.4))],
    "brow": [(0, 0.0), (A_SCARCITY + 14, 0.5), (A_LIMITED + 6, 0.8), (A_MAKE, 0.2), (150, 0.0)],
    "raise_": [(0, 0.0), (A_SCARCITY, 0.5), (A_MAKE, 0.8), (150, 1.0)],
    "wide": [(0, 0.0), (A_SCARCITY + 10, 0.4), (A_MAKE, 0.9), (150, 1.0)],
    "mouth": [(0, "flat"), (A_SCARCITY + 8, "o"), (A_LIMITED + 10, "flat"), (A_MAKE, "o"), (A_SEEM + 10, "smile"), (150, "smile")],
    "lean": [(0, 0), (A_MAKE, 0), (A_DESIR, 7), (150, 9)],
})


def orb_pos(f: float) -> tuple[float, float]:
    rise = smooth((f - (A_MAKE + 4)) / 40.0) * 34.0
    return 720.0, SHELF_Y - ORB_R - 6 - rise


def pose_p(f: float) -> dict:
    p = P_TL.at(f)
    p["lean"] += 0.7 * math.sin(f * 0.15)
    reach = smooth((f - (A_SEEM - 6)) / 22.0)
    if reach > 0:                                         # both hands toward the orb
        base = solve({**p, "handR": HANG_R, "handL": HANG_L}, ST, FLOOR)
        ox, oy = orb_pos(f)
        tR = (ox - 22 - base["shR"][0], oy + 12 - base["shR"][1])
        tL = (ox + 22 - base["shL"][0], oy + 24 - base["shL"][1])
        p["handR"] = (HANG_R[0] + (tR[0] - HANG_R[0]) * reach, HANG_R[1] + (tR[1] - HANG_R[1]) * reach)
        p["handL"] = (HANG_L[0] + (tL[0] - HANG_L[0]) * reach, HANG_L[1] + (tL[1] - HANG_L[1]) * reach)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (14, 70, 112))
    # footwork while approaching
    ww = smooth((f - (A_CAN + 8)) / 4.0) * (1 - smooth((f - (A_DESIR - 6)) / 8.0))
    if ww > 0.01:
        ph = (f - A_CAN) * 0.55
        p["footL"] = (-26 + 50 * ww * math.sin(ph), 347 - 30 * ww * max(0.0, math.cos(ph)))
        p["footR"] = (26 - 50 * ww * math.sin(ph), 347 - 30 * ww * max(0.0, -math.cos(ph)))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR, 1.0, {"rim": "right", "wall_shadow": False, "vis": 0.85})]


def kw(f: float) -> float:
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21))
    return k * (1 - 0.65 * smooth((f - A_SCARCITY) / 40.0))


def kspot(f: float) -> float:
    return (0.25 * smooth(f / 8.0) + 1.0 * smooth((f - (A_SCARCITY + 20)) / 36.0) + 0.45 * smooth((f - A_MAKE) / 40.0)) * (0.97 + 0.03 * math.sin(f * 0.43))


def lights(f: float):
    return [kw(f), kspot(f)]


def draw_orb(frame, cx, cy, r, glow_k, alpha=1.0):
    fxlib.glow_disc(frame, S, cx, cy, r * 1.6, RED, 0.55 * glow_k * alpha)
    fxlib.paint_ellipse(frame, S, cx, cy, r, r, RED * 0.88, alpha=0.96 * alpha)
    fxlib.paint_ellipse(frame, S, cx + r * 0.1, cy + r * 0.16, r * 0.72, r * 0.72, RED * 0.35, alpha=0.55 * alpha)          # shaded lower half
    fxlib.paint_ellipse(frame, S, cx - r * 0.36, cy - r * 0.38, r * 0.30, r * 0.20, WHITE, alpha=0.75 * alpha)                  # specular highlight
    fxlib.paint_ellipse(frame, S, cx, cy, r, r, WHITE * 0.8, alpha=0.30 * alpha, outline=2.5)


def pre_fx(frame, f, ks):
    # shelf: a plank with a lit top edge and two brackets
    fxlib.paint_rect(frame, S, 440, SHELF_Y, 1010, SHELF_Y + 16, (0.07, 0.065, 0.075), alpha=0.95)
    fxlib.paint_rect(frame, S, 440, SHELF_Y, 1010, SHELF_Y + 3, RED * 0.9, alpha=0.6 * max(ks[0], 0.3), mode="add", glow=4, glow_gain=0.6)
    for bx in (470, 980):
        fxlib.paint_poly(frame, S, [(bx - 14, SHELF_Y + 16), (bx + 14, SHELF_Y + 16), (bx, SHELF_Y + 70)], (0.06, 0.055, 0.065), alpha=0.95)
    for i, (x, kind) in enumerate(ORBS):
        if kind == "last":
            ox, oy = orb_pos(f)
            beat = 0.5 + 0.5 * math.sin(f * (0.30 + 0.35 * smooth((f - A_MAKE) / 40.0)))
            draw_orb(frame, ox, oy, ORB_R * (1.0 + 0.10 * beat * smooth((f - A_LIMITED) / 20.0)), 0.7 + 0.9 * beat * smooth((f - A_LIMITED) / 20.0) + 0.7 * smooth((f - A_MAKE) / 40.0))
            continue
        tv = VANISH[i if i < 4 else 3] if kind == "gone" else 0
        t = (f - tv) / 10.0
        if t >= 1.0:
            continue
        oy = SHELF_Y - ORB_R - 6
        if t < 0:
            draw_orb(frame, x, oy, ORB_R, 0.55)
        else:
            draw_orb(frame, x, oy - 10 * t, ORB_R * (1 + 0.5 * t), 0.55 * (1 - t), alpha=1 - t)
            fxlib.paint_ellipse(frame, S, x, oy - 10 * t, ORB_R * (1 + 2.4 * t), ORB_R * (1 + 2.4 * t), RED, alpha=0.8 * (1 - t), mode="add", outline=3, glow=5, glow_gain=0.6)
            for j in range(7):                                                      # sparks
                ang = j * 0.9 + x
                fxlib.glow_disc(frame, S, x + math.cos(ang) * 70 * t, oy - 10 * t + math.sin(ang) * 70 * t - 20 * t, 7, RED, 1.2 * (1 - t))


def fx(frame, f, acts, ks):
    # desire: red sparkles stream from the orb toward his eyes as it becomes irresistible
    d = smooth((f - A_MAKE) / 40.0)
    if d > 0:
        ox, oy = orb_pos(f)
        for j in range(5):
            ph = (f * 0.06 + j * 0.2) % 1.0
            fxlib.glow_disc(frame, S, ox + (acts[0][1]["hip_x"] + 20 - ox) * ph, oy + (780 - oy) * ph - 40 * math.sin(ph * math.pi), 6, RED, 0.9 * d * (1 - ph))


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.16, pan_start=(0.0, 0.0), pan_end=(0.8, 0.0), ease="ease_in_out"))


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
    WORLD = common.make_world_image(stage, ST, actors, lights, pre_fx=pre_fx, fx=fx)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        common.stills(WORLD, R, a.stills, "s06"); return
    ok = common.run(6, build_scene, WORLD, R, preview_f=A_AVAIL + 6, key_f=A_MORE + 6, contact_fs=[6, 26, 40, 58, 84, 108, 128, 146], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
