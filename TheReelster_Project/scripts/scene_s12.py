#!/usr/bin/env python3
"""Scene S12 - THE REELSTER OUTRO.  No narration. On-screen text: WHO'S REALLY CONTROLLING YOUR MIND? / THE REELSTER.

Continues S11: the strings that held him are reeled back up into the dark, he sinks back to the floor and looks up, then straight
at the viewer while the question appears word by word (MIND? in red) and slow red rings pulse out from his head. The brand
closes the video. The first 11 frames still carry "you have none." from S11. Longer fade-out: this is the end of the video.
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
import scene_s11 as s11
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, Scene, SceneRenderer
from script_plan import load_scene
from stage import Stage
from stickman import Style, Timeline, skeleton_bounds, smooth, solve

cfg = load_config()
ST = Style.from_cfg(cfg)
S = ST.world_scale
FLOOR = 1340.0
RED = np.asarray(cfg.rgb01("accent_red"), np.float32)
PLAN = load_scene(12)
assert PLAN.id == "S12" and PLAN.narration is None
BASE = s11.pose_p(149.0)                       # his last pose in S11: lifted, arms pulled up
BASE_LIFT = s11.lift(149.0)
TITLE = "WHO'S REALLY CONTROLLING YOUR MIND?"
TITLE_WORDS = [16, 24, 32, 42, 50]             # frame each title word appears (silent scene: designed, not audio-aligned)
BRAND_WORDS = [70, 80]
HANG_R, HANG_L = (34.0, 232.0), (-34.0, 232.0)
RETRACT0, RETRACT1 = 4, 40


def lift(f: float) -> float:
    return BASE_LIFT * (1 - smooth((f - 2) / 26.0))


def pose_p(f: float) -> dict:
    tl = Timeline({
        "head": [(0, BASE["head"]), (30, -8), (56, -6), (74, 0), (150, 0)],
        "gaze": [(0, BASE["gaze"]), (30, (0.0, -0.9)), (56, (0.0, -0.3)), (74, (0.0, 0.0)), (150, (0.0, 0.0))],
        "lean": [(0, BASE["lean"]), (30, 0), (150, 0)],
        "brow": [(0, 0.2), (56, 0.5), (150, 0.4)],
        "raise_": [(0, 0.0), (56, 0.4), (150, 0.3)],
        "wide": [(0, 0.8), (56, 0.5), (150, 0.4)],
        "mouth": [(0, "flat"), (150, "flat")],
    })
    p = tl.at(f)
    p["turn"], p["hip_x"] = 0.0, s11.XP + 0.8 * math.sin(f * 0.07)
    r = smooth((f - 6) / 26.0)
    p["handR"] = (BASE["handR"][0] + (HANG_R[0] - BASE["handR"][0]) * r, BASE["handR"][1] + (HANG_R[1] - BASE["handR"][1]) * r)
    p["handL"] = (BASE["handL"][0] + (HANG_L[0] - BASE["handL"][0]) * r, BASE["handL"][1] + (HANG_L[1] - BASE["handL"][1]) * r)
    p["footL"] = (BASE["footL"][0] + (-26.0 - BASE["footL"][0]) * r, BASE["footL"][1] + (347.0 - BASE["footL"][1]) * r)
    p["footR"] = (BASE["footR"][0] + (26.0 - BASE["footR"][0]) * r, BASE["footR"][1] + (347.0 - BASE["footR"][1]) * r)
    p["lean"] += 0.6 * math.sin(f * 0.15)
    p["blink"] = max(max(0.0, 1 - abs(f - c) / 2.0) for c in (40, 90, 128))
    return p


def actors(f: float):
    return [("P", pose_p(f), FLOOR - lift(f), 1.0, {"rim": "top", "wall_shadow": False})]


def lights(f: float):
    k = smooth(f / 4.0) * (0.92 + 0.12 * smooth((f - 20) / 60.0) + 0.03 * math.sin(f * 0.21))
    return [k, k * 0.7]


def fx(frame, f, acts, ks):
    pts = solve(acts[0][1], ST, FLOOR - lift(f))
    # the strings are reeled back up into the dark
    ret = smooth((f - RETRACT0) / (RETRACT1 - RETRACT0))
    if ret < 1.0:
        hx, hy = pts["head"]
        att = [(hx, hy - ST.head_r), pts["shL"], pts["shR"], pts["haL"], pts["haR"], pts["knL"], pts["knR"]]
        for (ax, ay), sx in zip(att, s11.STRING_X):
            ey = ay - (ay + 40) * ret
            fxlib.paint_line(frame, S, [(sx, -40), ((sx + ax) / 2, (ey - 40) / 2 + 12), (ax + (sx - ax) * ret, ey)], 3.2, RED, alpha=0.9 * (1 - ret), mode="add", glow=4, glow_gain=0.7)
    # slow rings from his head: the question hangs in the air
    hx, hy = pts["head"]
    for k in range(3):
        age = (f - 44 - k * 38) % 114
        if f >= 44 + k * 38 and age < 70:
            fxlib.paint_ellipse(frame, S, hx, hy, 80 + age * 5.0, 80 + age * 5.0, RED, alpha=0.38 * math.exp(-age / 34.0), mode="add", outline=3, glow=5, glow_gain=0.6)
    # a red underline sweeps under the brand
    u = smooth((f - 76) / 20.0)
    if u > 0:
        fxlib.paint_line(frame, S, [(540 - 200 * u, 1650), (540 + 200 * u, 1650)], 4, RED, alpha=0.8, mode="add", glow=6, glow_gain=0.9)


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=True) for c in PLAN.captions]
    caps.append(CaptionEvent(CaptionGroup(TITLE, ("MIND",)), TITLE_WORDS[0], 150, kind="headline", position="upper", word_starts=TITLE_WORDS))
    caps.append(CaptionEvent(CaptionGroup("THE REELSTER", ("REELSTER",)), BRAND_WORDS[0], 150, kind="caption", position="lower", word_starts=BRAND_WORDS))
    return Scene(source=lambda f, n: WORLD(f), captions=caps, fade_out=14,
                 camera=Camera(zoom_start=1.00, zoom_end=1.06, ease="ease_in_out"))


def layout_check() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        x0, y0, x1, y1 = skeleton_bounds(pose_p(f), ST, FLOOR - lift(f))
        lo = [min(lo[0], x0), min(lo[1], y0)]; hi = [max(hi[0], x1), max(hi[1], y1)]
    cap = cap_layout()
    return {"bounds": [round(v) for v in (*lo, *hi)], "title": cap, "ok": lo[0] >= l and hi[0] <= r and hi[1] <= 1447 and cap["clear_of_figure"]}


def cap_layout() -> dict:
    from caption_renderer import CaptionRenderer
    R = CaptionRenderer(cfg)
    o = R.build(CaptionGroup(TITLE, ("MIND",)), "headline", "upper")
    b = R.build(CaptionGroup("THE REELSTER", ("REELSTER",)), "caption", "lower")
    head_top = min(skeleton_bounds(pose_p(f), ST, FLOOR - lift(f))[1] for f in range(150))
    return {"title_lines": o.lines, "title_px": o.font_px, "title_bottom": o.ink_bbox[3], "head_top": round(head_top), "brand": b.lines,
            "clear_of_figure": o.ink_bbox[3] < head_top - 4}


WORLD = None


def main():
    global WORLD
    ap = argparse.ArgumentParser(); ap.add_argument("--stills", nargs="*", type=int)
    a = ap.parse_args()
    from stage import Lamp
    stage = Stage(cfg, ST, s11.LAMPS, floor=FLOOR, door=None)
    WORLD = common.make_world_image(stage, ST, actors, lights, fx=fx)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        common.stills(WORLD, R, a.stills, "s12"); return
    ok = common.run(12, build_scene, WORLD, R, preview_f=96, key_f=100, contact_fs=[4, 24, 44, 60, 80, 104, 128, 146], layout_fn=layout_check)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
