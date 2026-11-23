#!/usr/bin/env python3
"""Scene S08 - FOOT-IN-THE-DOOR.  "The foot-in-the-door effect starts with a small request."

A door stands ajar with red light spilling out. As it swings shut ("The foot-in-the-door effect") the stickman jams his foot into
the gap and the door stops on his shoe. On "a small request" he slides a small card through the slit; a hand from inside takes it
and the door eases open a little wider. Timing from the approved script + real voiceover.
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
PLAN = load_scene(8)
assert PLAN.id == "S08"
FR = {w["word"].strip(",:.'").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_THE, A_FOOT, A_EFFECT, A_STARTS, A_WITH, A_SMALL, A_REQ = (FR[k][0] for k in ("the", "foot-in-the-door", "effect", "starts", "with", "small", "request"))
LAMPS = [Lamp(pos=(300, 40), axis=(360, 1000), sigma=0.30, gain=1.4, reach=1000, halo=230),
         Lamp(pos=(700, 40), axis=(700, 700), sigma=0.50, gain=1.0, reach=1000, halo=230)]
HINGE_X, W_DOOR, TOP, BOT = 830.0, 310.0, 430.0, FLOOR
FRAME_L = HINGE_X - W_DOOR                          # left edge of the doorway
CARD = (466.0, 905.0)                               # where the small request is held out
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
FOOT_X = FRAME_L + 26.0
JAM_T = A_FOOT + 20
THETA_JAM = 29.0


def theta(f: float) -> float:
    """Door opening angle in degrees: ajar -> swings shut -> jams on his foot -> eases open when the request is taken."""
    t = 50.0
    t += (THETA_JAM + 1 - 50.0) * smooth((f - (A_FOOT + 4)) / 14.0)
    jolt = 2.5 * math.exp(-((f - JAM_T) / 2.0) ** 2) * math.sin((f - JAM_T) * 2.5)
    t += (THETA_JAM - (THETA_JAM + 1)) * smooth((f - (JAM_T - 2)) / 3.0) + jolt
    t += 11.0 * smooth((f - (A_SMALL + 14)) / 26.0)
    return t


def pose_p(f: float) -> dict:
    tl = Timeline({
        "hip_x": [(0, 190), (A_FOOT, 250), (JAM_T, 352), (150, 362)],
        "turn": [(0, 62), (150, 62)],
        "head": [(0, 4), (JAM_T, 0), (A_SMALL, -4), (150, -5)],
        "gaze": [(0, (1.0, 0.0)), (150, (1.0, 0.1))],
        "brow": [(0, 0.0), (A_FOOT, 0.3), (JAM_T, 0.6), (A_SMALL, 0.2), (150, 0.0)],
        "raise_": [(0, 0.3), (A_SMALL, 0.6), (150, 0.8)],
        "mouth": [(0, "flat"), (A_FOOT + 6, "o"), (JAM_T + 6, "flat"), (A_SMALL, "smile"), (150, "smile")],
        "lean": [(0, 2), (JAM_T - 6, 8), (A_SMALL, 10), (150, 8)],
        "wide": [(0, 0.0), (JAM_T, 0.5), (150, 0.2)],
    })
    p = tl.at(f)
    p["lean"] += 0.6 * math.sin(f * 0.15)
    # walk up, then plant the right foot in the gap
    wlk = smooth((f - 4) / 4.0) * (1 - smooth((f - (JAM_T - 4)) / 6.0))
    ph = f * 0.58
    footL = (-26 + 50 * wlk * math.sin(ph), 347 - 30 * wlk * max(0.0, math.cos(ph)))
    footR = (26 - 50 * wlk * math.sin(ph), 347 - 30 * wlk * max(0.0, -math.cos(ph)))
    plant = smooth((f - (A_FOOT + 2)) / 12.0)
    base = solve({**p, "footL": (-26, 347), "footR": (26, 347)}, ST, FLOOR)
    tx = FOOT_X - base["hipR"][0]
    footR = (footR[0] + (tx - footR[0]) * plant, footR[1] + (344 - footR[1]) * plant)
    p["footL"], p["footR"] = footL, footR
    # hands: swing while walking, then hold the small card out
    swing = 30 * wlk * math.sin(ph)
    hR, hL = (HANG_R[0] - swing, HANG_R[1]), (HANG_L[0] + swing, HANG_L[1])
    offer = smooth((f - (A_STARTS + 4)) / 10.0) * (1 - smooth((f - (A_SMALL + 26)) / 8.0))
    if offer > 0:
        b2 = solve({**p, "handR": HANG_R, "handL": HANG_L}, ST, FLOOR)
        tR = (CARD[0] - b2["shR"][0], CARD[1] - b2["shR"][1])
        hR = (hR[0] + (tR[0] - hR[0]) * offer, hR[1] + (tR[1] - hR[1]) * offer)
    p["handR"], p["handL"] = hR, hL
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (16, 60, 112))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR, 1.0, {"rim": "top", "wall_shadow": False})]


def lights(f: float):
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21))
    return [k, k * (0.8 + 0.25 * smooth((f - JAM_T) / 30.0))]


def door_quad(th: float):
    c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
    xf = HINGE_X - W_DOOR * c
    grow = 1.0 + 0.09 * s
    top_f, bot_f = TOP - (BOT - TOP) * (grow - 1) * 0.35, BOT + (BOT - TOP) * (grow - 1) * 0.1
    return xf, top_f, bot_f


def pre_fx(frame, f, ks):
    th = theta(f)
    xf, top_f, bot_f = door_quad(th)
    open_amt = 1.0 - math.cos(math.radians(th))
    # frame + glowing interior seen through the doorway (behind the door leaf)
    fxlib.paint_rect(frame, S, FRAME_L - 18, TOP - 20, HINGE_X + 22, BOT, (0.055, 0.05, 0.06), alpha=0.95)
    fxlib.paint_rect(frame, S, FRAME_L, TOP, HINGE_X, BOT, RED * 0.9, alpha=0.55 + 0.35 * open_amt, mode="add")
    fxlib.glow_disc(frame, S, FRAME_L + 40, 880, 330, RED, 0.35 + 0.5 * open_amt)
    # light spilling across the floor toward him
    fxlib.paint_poly(frame, S, [(FRAME_L, BOT), (xf, BOT), (xf - 260 - 380 * open_amt, 1590), (FRAME_L - 360 - 200 * open_amt, 1590)], RED, alpha=0.30 + 0.35 * open_amt, mode="add")
    # the door leaf
    leaf = [(HINGE_X, TOP), (xf, top_f), (xf, bot_f), (HINGE_X, BOT)]
    fxlib.paint_poly(frame, S, leaf, (0.075, 0.068, 0.078), alpha=0.97)
    for (a, b) in ((0.12, 0.40), (0.60, 0.92)):                                                # inset panels
        pts = []
        for u, v in ((0.12, a), (0.88, a), (0.88, b), (0.12, b)):
            x = HINGE_X + (xf - HINGE_X) * u
            ytop = TOP + (top_f - TOP) * u
            ybot = BOT + (bot_f - BOT) * u
            pts.append((x, ytop + (ybot - ytop) * v))
        fxlib.paint_line(frame, S, pts + [pts[0]], 2.5, (0.11, 0.10, 0.115), alpha=0.9)
    fxlib.paint_line(frame, S, [(xf, top_f), (xf, bot_f)], 4, RED * 0.95, alpha=0.8, mode="add", glow=5, glow_gain=0.9)   # lit free edge
    fxlib.paint_ellipse(frame, S, xf + 26, 880, 9, 9, (0.5, 0.45, 0.4), alpha=0.9)                                          # handle
    # a hand from inside takes the card through the slit
    t0 = A_SMALL + 10
    if t0 <= f < t0 + 44:
        e = smooth((f - t0) / 12.0) * (1 - smooth((f - (t0 + 28)) / 14.0))
        hx = xf - 10 - (xf - 40 - CARD[0]) * e
        fxlib.paint_line(frame, S, [(xf + 80, 905), (hx, 905)], 24, (0.02, 0.018, 0.022), alpha=0.97)
        fxlib.paint_ellipse(frame, S, hx - 8, 905, 20, 16, (0.02, 0.018, 0.022), alpha=0.97)
        fxlib.paint_line(frame, S, [(hx - 20, 905), (hx, 905)], 24, RED, alpha=0.55 * e, mode="add", glow=4, glow_gain=0.6)


def fx(frame, f, acts, ks):
    # the small request: a little card in his hand, handed over and gone
    pts = solve(acts[0][1], ST, FLOOR)
    if A_STARTS + 4 <= f < A_SMALL + 24:
        hx, hy = pts["haR"]
        a = smooth((f - (A_STARTS + 4)) / 6.0)
        if f > A_SMALL + 14:
            a *= 1 - smooth((f - (A_SMALL + 14)) / 8.0)
        cx = hx + 8 + (1 - smooth((f - (A_SMALL + 12)) / 12.0)) * 0
        fxlib.paint_rect(frame, S, cx - 4, hy - 20, cx + 40, hy + 10, WHITE, alpha=0.95 * a, radius=3)
        fxlib.paint_rect(frame, S, cx - 4, hy - 20, cx + 40, hy + 10, RED, alpha=0.9 * a, mode="add", radius=3, outline=2.5, glow=4, glow_gain=0.7)
        fxlib.glow_disc(frame, S, cx + 18, hy - 5, 28, RED, 0.5 * a)
    # impact ring where the door lands on his shoe
    t = f - JAM_T
    if 0 <= t < 14:
        fxlib.paint_ellipse(frame, S, FOOT_X + 20, FLOOR - 14, 14 + t * 5, 8 + t * 2.4, RED, alpha=0.9 * math.exp(-t / 5.0), mode="add", outline=3, glow=4, glow_gain=0.8)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.14, pan_start=(0.0, 0.0), pan_end=(0.5, 0.2), ease="ease_in_out"))


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
        common.stills(WORLD, R, a.stills, "s08"); return
    ok = common.run(8, build_scene, WORLD, R, preview_f=A_SMALL + 8, key_f=A_SMALL + 12, contact_fs=[6, 26, 44, 56, 72, 90, 110, 140], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
