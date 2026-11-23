#!/usr/bin/env python3
"""Scene S07 - URGENCY.  "That's why countdowns and 'only a few left' can influence decisions."

A red countdown timer starts ticking on the wall ("countdowns"); a hanging sign counts 3 -> 2 -> 1 LEFT as the stock on a pedestal
vanishes ("only a few left"); the stickman sweats and panics, then - "can influence decisions" - runs over and grabs the last one
as the timer reaches 00:00. Timing from the approved script + real voiceover.
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
PLAN = load_scene(7)
assert PLAN.id == "S07"
FR = {w["word"].strip(",:.'").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_COUNT, A_ONLY, A_FEW, A_LEFT, A_CAN, A_INFL, A_DEC = (FR[k][0] for k in ("countdowns", "only", "few", "left", "can", "influence", "decisions"))
LAMPS = [Lamp(pos=(540, 40), axis=(540, 600), sigma=0.55, gain=1.3, reach=1000, halo=260),
         Lamp(pos=(560, 40), axis=(520, 1000), sigma=0.34, gain=1.3, reach=1100, halo=200)]
PED = (740.0, 1070.0, 900.0)                        # pedestal x0, top y, x1
ORB_XY = [(775.0, 1042.0), (820.0, 1042.0), (865.0, 1042.0)]
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
GRAB = A_INFL + 18


def seconds_left(f: float) -> int:
    return max(0, 9 - int((f - A_COUNT) / 10.5)) if f >= A_COUNT else 9


def stock(f: float) -> int:
    return 3 - (f >= A_FEW) - (f >= A_LEFT)


def pose_p(f: float) -> dict:
    panic = smooth((f - A_COUNT) / 20.0) * (1 - smooth((f - A_CAN) / 8.0))
    shake = math.sin(f * 1.7) * 2.0 * panic
    tl = Timeline({
        "hip_x": [(0, 300), (A_INFL - 8, 300), (A_INFL + 14, 640), (150, 648)],
        "turn": [(0, 25), (A_INFL - 6, 25), (A_INFL + 2, 75), (150, 70)],
        "head": [(0, 0), (A_COUNT, -12), (A_CAN, -12), (A_INFL + 6, 4), (150, 6)],
        "gaze": [(0, (0.0, -0.3)), (A_COUNT, (0.0, -1.0)), (A_CAN, (0.4, -0.8)), (A_INFL + 8, (1.0, 0.2)), (150, (1.0, 0.3))],
        "brow": [(0, 0.0), (A_COUNT + 6, 0.8), (A_CAN, 1.0), (GRAB, 0.3), (150, 0.0)],
        "raise_": [(0, 0.2), (A_COUNT + 6, 0.8), (A_CAN, 0.9), (150, 0.4)],
        "wide": [(0, 0.0), (A_COUNT + 6, 0.8), (A_CAN, 1.0), (GRAB, 0.6), (150, 0.4)],
        "mouth": [(0, "flat"), (A_COUNT + 8, "o"), (A_CAN, "wavy"), (GRAB, "smile"), (150, "smile")],
        "lean": [(0, 0), (A_COUNT, 3), (A_CAN, 6), (A_INFL + 10, 12), (GRAB, 8), (150, 0)],
    })
    p = tl.at(f)
    p["hip_x"] += shake
    p["lean"] += 0.7 * math.sin(f * 0.15)
    # hands: to the head while panicking, pumping while running, then reaching for the last orb, then clutching it
    head_hi, head_lo = (40.0, -96.0), (-40.0, -96.0)
    hR = (HANG_R[0] + (head_hi[0] - HANG_R[0]) * panic, HANG_R[1] + (head_hi[1] - HANG_R[1]) * panic)
    hL = (HANG_L[0] + (head_lo[0] - HANG_L[0]) * panic, HANG_L[1] + (head_lo[1] - HANG_L[1]) * panic)
    run = smooth((f - (A_INFL - 6)) / 4.0) * (1 - smooth((f - (GRAB - 8)) / 6.0))
    if run > 0.01:
        ph = (f - (A_INFL - 6)) * 0.85
        hR = (hR[0] * (1 - run) + (30 + 55 * math.sin(ph)) * run, hR[1] * (1 - run) + (170 + 0 * math.cos(ph)) * run)
        hL = (hL[0] * (1 - run) + (-30 - 55 * math.sin(ph)) * run, hL[1] * (1 - run) + 170 * run)
        p["footL"] = (-26 + 62 * run * math.sin(ph), 347 - 42 * run * max(0.0, math.cos(ph)))
        p["footR"] = (26 - 62 * run * math.sin(ph), 347 - 42 * run * max(0.0, -math.cos(ph)))
    p["handR"], p["handL"] = hR, hL
    reach = smooth((f - (GRAB - 8)) / 8.0)
    if reach > 0:
        base = solve({**p, "handR": HANG_R, "handL": HANG_L}, ST, FLOOR)
        ox, oy = ORB_XY[0][0] - 20, ORB_XY[0][1]
        tR = (ox - base["shR"][0], oy - base["shR"][1])
        p["handR"] = (p["handR"][0] + (tR[0] - p["handR"][0]) * reach, p["handR"][1] + (tR[1] - p["handR"][1]) * reach)
        clutch = smooth((f - (GRAB + 8)) / 8.0)                                       # pulls it to his chest
        if clutch > 0:
            p["handR"] = (p["handR"][0] + (-30 - p["handR"][0]) * clutch, p["handR"][1] + (60 - p["handR"][1]) * clutch)
            p["handL"] = (p["handL"][0] + (30 - p["handL"][0]) * clutch, p["handL"][1] + (70 - p["handL"][1]) * clutch)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (12, 56))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR, 1.0, {"rim": "top", "wall_shadow": False})]


def lights(f: float):
    alarm = 1.0 + 0.25 * max(0.0, math.sin(f * 0.9)) * smooth((f - A_CAN) / 6.0) * (1 - smooth((f - (GRAB + 18)) / 8.0))     # pulses once the timer is nearly out
    k = smooth(f / 8.0) * (0.95 + 0.04 * math.sin(f * 0.21)) * alarm
    return [k * (0.55 + 0.45 * smooth((f - A_COUNT + 6) / 10.0)), k * 0.9]


def orb(frame, cx, cy, r, alpha, k=1.0):
    fxlib.glow_disc(frame, S, cx, cy, r * 1.6, RED, 0.5 * k * alpha)
    fxlib.paint_ellipse(frame, S, cx, cy, r, r, RED * 0.88, alpha=0.96 * alpha)
    fxlib.paint_ellipse(frame, S, cx - r * 0.36, cy - r * 0.38, r * 0.3, r * 0.2, WHITE, alpha=0.7 * alpha)
    fxlib.paint_ellipse(frame, S, cx, cy, r, r, WHITE * 0.8, alpha=0.28 * alpha, outline=2)


def pre_fx(frame, f, ks):
    # ---- the countdown timer (appears on "countdowns")
    a = smooth((f - (A_COUNT - 4)) / 8.0)
    if a > 0:
        sec = seconds_left(f)
        low = sec <= 3 and f >= A_COUNT
        jit = (math.sin(f * 2.3) * 3.0, math.cos(f * 2.9) * 2.0) if low else (0.0, 0.0)
        x0, y0, x1, y1 = 330 + jit[0], 290 + jit[1], 810 + jit[0], 450 + jit[1]
        fxlib.paint_rect(frame, S, x0, y0, x1, y1, (0.03, 0.03, 0.04), alpha=0.88 * a, radius=14)
        fxlib.paint_rect(frame, S, x0, y0, x1, y1, RED, alpha=0.7 * a, mode="add", radius=14, outline=4, glow=8, glow_gain=0.8)
        flash = 1.0 if sec > 0 else (0.5 + 0.5 * math.sin(f * 1.3))
        txt = f"00:0{sec}" if (sec > 0 or int(f / 6) % 2 == 0) else "00:00"
        fxlib.paint_text(frame, S, cfg, txt, (x0 + x1) / 2, (y0 + y1) / 2, 118, RED * 1.1, alpha=a * flash, mode="add", glow=10, glow_gain=1.0)
        fxlib.glow_disc(frame, S, (x0 + x1) / 2, (y0 + y1) / 2, 240, RED, 0.12 * a * flash)
    # ---- hanging sign "ONLY n LEFT"
    s_a = smooth((f - (A_ONLY - 6)) / 8.0)
    if s_a > 0:
        n = stock(f)
        flip = 1.0 - min(1.0, abs(f - A_FEW) / 4.0) if abs(f - A_FEW) < 4 else (1.0 - min(1.0, abs(f - A_LEFT) / 4.0) if abs(f - A_LEFT) < 4 else 0.0)
        for cx in (430, 710):
            fxlib.paint_line(frame, S, [(cx, 0), (cx, 478)], 3, (0.12, 0.11, 0.12), alpha=0.9 * s_a)
        fxlib.paint_rect(frame, S, 360, 478, 780, 562, (0.05, 0.045, 0.05), alpha=0.92 * s_a, radius=10)
        fxlib.paint_rect(frame, S, 360, 478, 780, 562, WHITE * 0.7, alpha=0.5 * s_a, mode="add", radius=10, outline=3)
        fxlib.paint_text(frame, S, cfg, f"ONLY {n} LEFT", 570, 520, 54, (RED * 1.15 if n <= 1 else WHITE), alpha=s_a * (1 - 0.8 * flip), mode="add" if n <= 1 else "over",
                         glow=6 if n <= 1 else 0, glow_gain=0.9 if n <= 1 else 0)
    # ---- pedestal + stock
    fxlib.paint_rect(frame, S, PED[0], PED[1], PED[2], FLOOR, (0.06, 0.055, 0.065), alpha=0.96)
    fxlib.paint_rect(frame, S, PED[0], PED[1], PED[2], PED[1] + 4, RED * 0.9, alpha=0.7, mode="add", glow=4, glow_gain=0.7)
    for i, (x, y) in enumerate(ORB_XY):
        # i=0 is the last one left (grabbed at GRAB); i=1 vanishes at "few", i=2 at "left"
        tv = {1: A_FEW, 2: A_LEFT}.get(i)
        if i == 0:
            if f >= GRAB + 4:
                continue
            orb(frame, x, y - 2 * math.sin(f * 0.3), 20, 1.0, 0.8 + 0.5 * math.sin(f * 0.4))
        else:
            t = (f - tv) / 8.0
            if t >= 1.0:
                continue
            if t < 0:
                orb(frame, x, y, 20, 1.0)
            else:
                orb(frame, x, y - 8 * t, 20 * (1 + 0.4 * t), 1 - t)
                fxlib.paint_ellipse(frame, S, x, y - 8 * t, 20 * (1 + 2.2 * t), 20 * (1 + 2.2 * t), RED, alpha=0.8 * (1 - t), mode="add", outline=3, glow=4, glow_gain=0.6)


def fx(frame, f, acts, ks):
    # sweat drops fly off his head while he panics
    pose = acts[0][1]
    pts = solve(pose, ST, FLOOR)
    panic = smooth((f - A_COUNT) / 20.0) * (1 - smooth((f - A_CAN) / 8.0))
    if panic > 0.2:
        hx, hy = pts["head"]
        for j in range(3):
            age = (f * 0.7 + j * 9.0) % 24.0
            side = (-1) ** j
            fxlib.paint_ellipse(frame, S, hx + side * (ST.head_r * 0.9 + age * 1.4), hy - 10 + age * 2.6 + age * age * 0.12, 4.5, 7.0, BLUE * 1.3, alpha=0.9 * (1 - age / 24.0), mode="add")
    # the snap decision: a red flash ring when he grabs it
    t = f - GRAB
    if 0 <= t < 16:
        ox, oy = ORB_XY[0]
        fxlib.paint_ellipse(frame, S, ox, oy, 30 + t * 9, 30 + t * 9, RED, alpha=0.9 * math.exp(-t / 6.0), mode="add", outline=4, glow=8, glow_gain=0.9)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    return Scene(source=lambda f, n: WORLD(f), captions=caps,
                 camera=Camera(zoom_start=1.00, zoom_end=1.10, pan_start=(0.0, 0.0), pan_end=(0.4, 0.0), ease="ease_in_out"))


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
        common.stills(WORLD, R, a.stills, "s07"); return
    ok = common.run(7, build_scene, WORLD, R, preview_f=A_LEFT + 6, key_f=A_LEFT + 8, contact_fs=[6, 36, 62, 80, 96, 116, 132, 148], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
