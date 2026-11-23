#!/usr/bin/env python3
"""Scene S03 - THE EMOTIONAL EFFECT.   "That feeling can hurt, even when no one says a word."

Continues straight from S02's last frame: he is alone in the narrow red spotlight, head down, the group still walking
away into the dark. "That feeling": a hand goes to his chest. "can hurt": red pain ripples spread from the chest and he folds
in around it. The pause, then "even when no one says a word": silence - the spotlight flickers, a single tear forms and
falls to the floor. Slow camera push toward him, continuing S02's move.

Timing from the approved script + real voiceover (script_plan.py). The first 19 frames still carry "belonging." from S02.
Outputs: scenes/S03.mp4, assets/visuals/visual_03.png, qa/S03_preview.png, qa/S03_with_voice.mp4, qa/S03_contact.png
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa_scene
import scene_s02 as s2
import world_render
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import Camera, CaptionEvent, FFmpegWriter, Scene, SceneRenderer
from script_plan import load_scene
from stage import Stage
from stickman import Style, Timeline, skeleton_bounds, smooth, solve

cfg = load_config()
ST = Style.from_cfg(cfg)
S = ST.world_scale
FLOOR = s2.FLOOR
OFFSET = 150                         # S02 frames elapsed: group walk + lights continue from there
PLAN = load_scene(3)
assert PLAN.id == "S03" and PLAN.narration.startswith("That feeling")
FR = {w["word"].strip(",.").lower(): (w["start"], w["end"]) for w in PLAN.words}
A_THAT, A_FEEL, A_CAN, A_HURT = (FR[k][0] for k in ("that", "feeling", "can", "hurt"))
A_EVEN, A_NO, A_SAYS, A_WORD = (FR[k][0] for k in ("even", "no", "says", "word"))
BASE = s2.p_pose(149.0)              # his exact pose in S02's final frame (continuity)
RED = np.asarray(cfg.rgb01("accent_red"), np.float32)
BLUE = np.asarray(cfg.rgb01("research_blue"), np.float32)

# ================================================================ choreography
HANG_L, HANG_R = s2.HANG_L, s2.HANG_R
HEART_R, HEART_L = (-34.0, 62.0), (36.0, 66.0)          # hand targets relative to each shoulder: hands on his chest
TL3 = Timeline({
    "turn": [(0, BASE["turn"]), (A_HURT + 8, 28), (150, 24)],
    "head": [(0, BASE["head"]), (A_FEEL, BASE["head"]), (A_HURT + 4, 28), (A_NO + 6, 30), (150, 33)],
    "lean": [(0, BASE["lean"]), (A_CAN, BASE["lean"]), (A_HURT + 10, 15), (A_EVEN, 14), (150, 17)],
    "handR": [(0, BASE["handR"]), (A_THAT + 2, BASE["handR"]), (A_CAN - 2, HEART_R), (150, HEART_R)],
    "handL": [(0, BASE["handL"]), (A_HURT + 2, BASE["handL"]), (A_HURT + 14, HEART_L), (150, HEART_L)],
    "footL": [(0, (-26.0, 347.0)), (A_CAN, (-26.0, 347.0)), (A_HURT + 12, (-26.0, 318.0)), (150, (-26.0, 310.0))],
    "footR": [(0, (26.0, 347.0)), (A_CAN, (26.0, 347.0)), (A_HURT + 12, (26.0, 318.0)), (150, (26.0, 310.0))],
    "gaze": [(0, BASE["gaze"]), (150, (0.2, 1.0))],
    "brow": [(0, 1.0), (150, 1.0)],
    "raise_": [(0, 0.0), (150, 0.0)],
    "wide": [(0, BASE.get("wide", 0.1)), (A_THAT, 0.1), (A_HURT, 0.5), (A_EVEN, 0.15), (150, 0.1)],
    "mouth": [(0, "frown"), (A_HURT + 6, "wavy"), (A_EVEN + 8, "frown")],
})


def p_pose(f: float) -> dict:
    p = TL3.at(f)
    pain = smooth((f - A_HURT) / 6.0) * (1 - 0.5 * smooth((f - A_EVEN) / 20.0))
    p["hip_x"] = BASE["hip_x"] + 1.3 * pain * math.sin(f * 1.9)                    # trembling
    p["lean"] += 0.8 * math.sin(f * 0.16)                                            # breathing
    wince = 0.85 * smooth((f - (A_HURT + 2)) / 3.0) * (1 - smooth((f - (A_HURT + 16)) / 4.0))
    heavy = max(0.0, 0.9 * (1 - abs(f - (TEAR_BORN - 2)) / 3.0))                        # slow blink as the tear forms
    p["blink"] = max(wince, heavy)
    return p


def actors(f: float):
    out = [("P", p_pose(f), FLOOR, 1.0)]
    for i in range(4):
        pose, fl, sc = s2.member(i, f + OFFSET)
        if sc > 0.3 and pose["hip_x"] < 1250:            # skip figures that have walked out of the frame / into the distance
            out.append((f"G{i + 1}", pose, fl, sc))
    return out


def lights(f: float) -> list[float]:
    flick = 1 - 0.30 * math.exp(-(((f - (A_NO + 5)) / 7.0) ** 2)) - 0.20 * math.exp(-(((f - (A_SAYS + 14)) / 6.0) ** 2))
    return [s2.k_wide(f + OFFSET), s2.k_solo(f + OFFSET) * flick]


# ===================================================================== effects
RING_BIRTHS = [(A_HURT, 1.0), (A_HURT + 10, 0.85), (A_HURT + 22, 0.7), (A_EVEN - 4, 0.5), (A_NO, 0.35), (A_WORD - 4, 0.25)]
TEAR_BORN, TEAR_DROP, TEAR_LAND = A_SAYS - 4, A_SAYS + 6, A_WORD + 17     # lands at f138, before the scene's fade-out dip (f141)
LAND_Y = 1372.0


def _window(frame, cx: float, cy: float, R: float):
    """World-pixel window around (cx,cy) given in final px; returns slices + local coordinate grids in FINAL px."""
    H, W = frame.shape[1], frame.shape[2]
    x0, y0, x1, y1 = int(cx * S - R * S), int(cy * S - R * S), int(cx * S + R * S), int(cy * S + R * S)
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1c <= x0c or y1c <= y0c:
        return None
    yy, xx = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
    return (slice(y0c, y1c), slice(x0c, x1c)), (xx / S - cx), (yy / S - cy)


def fx(frame: np.ndarray, F: float, acts, ks):
    f = F - OFFSET
    P = next(a for a in acts if a[0] == "P")
    pose, fl = P[1], P[2]
    pts = solve(pose, ST, fl)
    hip, spine = pts["hip"], pts["spine"]
    heart = (hip[0] + 0.70 * (spine[0] - hip[0]) - 8, hip[1] + 0.70 * (spine[1] - hip[1]))
    k = ks[1]
    # ---- pain: glow + expanding ripples from the chest
    win = _window(frame, heart[0], heart[1], 340)
    if win is not None and f >= A_HURT - 8:
        sl, dx, dy = win
        d = np.sqrt(dx * dx + dy * dy)
        amp = smooth((f - (A_HURT - 6)) / 8.0) * (1 - 0.65 * smooth((f - A_EVEN) / 30.0))
        glow = np.exp(-((d / 46.0) ** 2)) * amp * (0.75 + 0.25 * math.sin(f * 0.9)) * 1.1
        rings = np.zeros_like(d)
        for tb, strength in RING_BIRTHS:
            age = f - tb
            if 0 <= age < 80:
                r = 14 + age * 6.2
                rings += np.exp(-(((d - r) / 3.4) ** 2)) * strength * math.exp(-age / 30.0)
        frame[:, sl[0], sl[1]] += (glow + rings * 1.15)[None] * RED[:, None, None] * min(1.0, 0.4 + k)
    # ---- the tear
    if f >= TEAR_BORN:
        hx, hy = pts["head"]
        ha = math.radians(pose["lean"] + pose["head"])
        c, s_ = math.cos(ha), math.sin(ha)
        cT, sT = math.cos(math.radians(pose["turn"])), math.sin(math.radians(pose["turn"]))
        u = 0.36 * max(cT, 0.28) + 0.62 * sT
        ex, ey = hx + (u * c + 0.06 * s_) * ST.head_r, hy + (u * s_ - 0.06 * c) * ST.head_r
        chin = (hx + 0.60 * ST.head_r * c, hy + 0.98 * ST.head_r)
        if f < TEAR_DROP:                                           # forms at the eye, slides down the cheek
            t = (f - TEAR_BORN) / (TEAR_DROP - TEAR_BORN)
            tx, ty, size, vel = lerp(ex, chin[0], smooth(t)), lerp(ey, chin[1], smooth(t)), min(1.0, 0.25 + 1.5 * t), 0.0
            x_rel = None
        elif f < TEAR_LAND:                                         # free fall
            t = f - TEAR_DROP
            g = 2 * (LAND_Y - chin[1]) / ((TEAR_LAND - TEAR_DROP) ** 2)
            tx, ty, size, vel = chin[0], chin[1] + 0.5 * g * t * t, 1.0, g * t
        else:
            tx = None
        if tx is not None:
            win = _window(frame, tx, ty, 60)
            if win is not None:
                sl, dx, dy = win
                sy = 7.0 + vel * 0.55                                # motion-stretched while falling
                core = np.exp(-((dx / 6.0) ** 2) - ((dy / (sy * 1.25)) ** 2)) * 1.9 * size
                halo = np.exp(-((dx / 22.0) ** 2) - ((dy / (sy * 2.6)) ** 2)) * 0.5 * size
                col = np.asarray([0.62, 0.80, 1.0], np.float32)
                frame[:, sl[0], sl[1]] += (core + halo)[None] * col[:, None, None]
        if f >= TEAR_LAND:                                          # splash ring on the floor
            win = _window(frame, chin[0], LAND_Y, 120)
            if win is not None:
                sl, dx, dy = win
                age = f - TEAR_LAND
                dd = np.sqrt(dx * dx + (dy * 3.4) ** 2)
                ring = np.exp(-(((dd - (3 + age * 5.0)) / 3.4) ** 2)) * math.exp(-age / 6.0) * 1.3
                frame[:, sl[0], sl[1]] += ring[None] * np.asarray([0.62, 0.80, 1.0], np.float32)[:, None, None]


def lerp(a, b, t):
    return a + (b - a) * t


# ================================================================== rendering
STAGE = None


def world_image(f: float) -> Image.Image:
    fr = world_render.render(STAGE, ST, cfg, f + OFFSET, actors(f), lights(f), fx=fx)
    return Image.fromarray((np.clip(fr, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8), "RGB")


def build_scene() -> Scene:
    caps = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                         word_starts=c.word_starts, fade_out=not c.carry_out) for c in PLAN.captions]
    # S02 ended at zoom 1.12 / pan -0.8: continue the same slow push toward him
    return Scene(source=lambda f, n: world_image(f), captions=caps,
                 camera=Camera(zoom_start=1.12, zoom_end=1.26, pan_start=(-0.8, 0.0), pan_end=(-1.0, 0.30), ease="ease_in_out"))


def check_layout() -> dict:
    l, t, r, b = cfg.safe_rect
    lo, hi = [1e9, 1e9], [-1e9, -1e9]
    for f in range(150):
        x0, y0, x1, y1 = skeleton_bounds(p_pose(f), ST, FLOOR)
        lo = [min(lo[0], x0), min(lo[1], y0)]; hi = [max(hi[0], x1), max(hi[1], y1)]
    ok = lo[0] >= l and hi[0] <= r and lo[1] >= t and hi[1] <= 1447
    return {"character_bounds": [round(v) for v in (*lo, *hi)], "safe": [l, t, r, 1447], "ok": ok}


def main():
    global STAGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", nargs="*", type=int)
    a = ap.parse_args()
    STAGE = Stage(cfg, ST, [s2.LAMP_GROUP, s2.LAMP_SOLO], floor=FLOOR)
    R = SceneRenderer(cfg)
    if a.stills is not None:
        for f in a.stills:
            s2.to_png(world_image(f), R, f, cfg.path(f"qa/s03_dev_{f:03d}.png"))
        return
    lay = check_layout()
    print("layout:", lay)
    if not lay["ok"]:
        sys.exit("character leaves the safe area")
    scene = build_scene()
    out = cfg.path("scenes/S03.mp4")
    with FFmpegWriter(out, cfg) as w:
        n = R.render_scene(scene, w)                   # strict: raises unless exactly 150 frames
    p = qa_scene.probe(out)
    sync = qa_scene.verify_caption_sync(out, scene.captions, cfg)
    ok = qa_scene.report(p, sync, None) and n == 150
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf", f"select=eq(n\\,{A_HURT + 12})",
                    "-frames:v", "1", str(cfg.path("qa/S03_preview.png"))], check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf",
                    "select='eq(n\\,6)+eq(n\\,36)+eq(n\\,56)+eq(n\\,76)+eq(n\\,102)+eq(n\\,118)+eq(n\\,134)+eq(n\\,148)',scale=270:480,tile=8x1",
                    "-frames:v", "1", str(cfg.path("qa/S03_contact.png"))], check=True)
    s2.to_png(world_image(A_HURT + 14), R, 0, cfg.path("assets/visuals/visual_03.png"))
    qa_scene.voice_preview(out, cfg.path("qa/S03_with_voice.mp4"), cfg, 3, PLAN.spill_out)
    print("S03 VERIFIED" if ok else "S03 FAILED VERIFICATION")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
