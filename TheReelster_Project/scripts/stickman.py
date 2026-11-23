"""Stickman character engine (one consistent design for the whole video).

* Rig: hip -> torso -> neck -> head, shoulder/hip bars, two-bone limbs solved with IK, so
  gestures are keyframed as *where the hand/foot goes* (px relative to shoulder/hip).
* Drawing: Pillow at supersample resolution, returned as float masks at world resolution:
      outline (light edge), body (black fill), face_w / face_k (white / black face features),
      crack (red head cracks, for the 'psychological weakness' effect).
* Coordinates are FINAL 1080x1920 pixels; `world_scale` maps them to the render canvas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from config import Config

Vec = tuple[float, float]


# ----------------------------------------------------------------- style
@dataclass
class Style:
    head_r: float; torso: float; neck: float; thigh: float; shin: float
    upper_arm: float; forearm: float; body_w: float; limb_w: float; hand_r: float
    foot: float; outline: float; shoulder_half: float; hip_half: float
    supersample: int; world_scale: float

    @classmethod
    def from_cfg(cls, cfg: Config) -> "Style":
        s = cfg.raw["framework"]["stickman"]
        vals = {k: s[k] for k in cls.__dataclass_fields__ if k != "outline"}
        return cls(outline=s["outline_px"], **vals)


# ------------------------------------------------------------- keyframes
def smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Timeline:
    """name -> [(frame, value)]; floats or tuples are smoothstep-interpolated, strings step."""

    def __init__(self, tracks: dict[str, list[tuple[float, object]]]):
        self.tracks = {k: sorted(v, key=lambda kv: kv[0]) for k, v in tracks.items()}

    def at(self, f: float) -> dict:
        out = {}
        for name, keys in self.tracks.items():
            if f <= keys[0][0]:
                out[name] = keys[0][1]; continue
            if f >= keys[-1][0]:
                out[name] = keys[-1][1]; continue
            for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
                if f0 <= f <= f1:
                    t = smooth((f - f0) / (f1 - f0)) if f1 > f0 else 1.0
                    if isinstance(v0, str):
                        out[name] = v0 if t < 1.0 else v1
                    elif isinstance(v0, tuple):
                        out[name] = tuple(a + (b - a) * t for a, b in zip(v0, v1))
                    else:
                        out[name] = v0 + (v1 - v0) * t
                    break
        return out


DEFAULT_POSE = dict(
    hip_x=540.0, lean=0.0, head=0.0, shrug=0.0,
    handL=(-34.0, 232.0), handR=(34.0, 232.0), footL=(-26.0, 347.0), footR=(26.0, 347.0),
    gaze=(0.0, 0.0), brow=0.0, raise_=0.0, wide=0.0, mouth="flat", blink=0.0,
)


# ------------------------------------------------------------------- IK
def ik_outward(root: Vec, target: Vec, l1: float, l2: float, side: int) -> tuple[Vec, Vec]:
    """Two-bone IK. Returns (joint, end). `side` -1 = left limb, +1 = right limb; the joint
    bends away from the body midline (elbows/knees point outward)."""
    rx, ry = root
    dx, dy = target[0] - rx, target[1] - ry
    d = max(min(math.hypot(dx, dy), l1 + l2 - 1e-3), abs(l1 - l2) + 1e-3)
    a = math.atan2(dy, dx)
    A = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    best = None
    for s in (+1, -1):
        ex, ey = rx + l1 * math.cos(a + s * A), ry + l1 * math.sin(a + s * A)
        score = side * (ex - rx) + 0.6 * (ey - ry)      # outward, and prefer elbows/knees low
        if best is None or score > best[0]:
            best = (score, (ex, ey))
    return best[1], (rx + d * math.cos(a), ry + d * math.sin(a))


def solve(pose: dict, st: Style, floor_y: float) -> dict[str, Vec]:
    """Forward kinematics + IK -> named joint positions, grounded so the lowest foot touches floor_y."""
    p = {**DEFAULT_POSE, **pose}
    lean = math.radians(p["lean"])
    up = (math.sin(lean), -math.cos(lean))
    perp = (math.cos(lean), math.sin(lean))
    hip = (p["hip_x"], 0.0)
    spine = (hip[0] + up[0] * st.torso, hip[1] + up[1] * st.torso)
    shc = (spine[0] - up[0] * 8, spine[1] - up[1] * 8 - p["shrug"] * 16)
    shL = (shc[0] - perp[0] * st.shoulder_half, shc[1] - perp[1] * st.shoulder_half)
    shR = (shc[0] + perp[0] * st.shoulder_half, shc[1] + perp[1] * st.shoulder_half)
    neck = (spine[0] + up[0] * st.neck, spine[1] + up[1] * st.neck)
    ha = math.radians(p["lean"] + p["head"])
    head = (neck[0] + math.sin(ha) * st.head_r * 0.92, neck[1] - math.cos(ha) * st.head_r * 0.92)
    hipL, hipR = (hip[0] - st.hip_half, 0.0), (hip[0] + st.hip_half, 0.0)

    elL, haL = ik_outward(shL, (shL[0] + p["handL"][0], shL[1] + p["handL"][1]), st.upper_arm, st.forearm, -1)
    elR, haR = ik_outward(shR, (shR[0] + p["handR"][0], shR[1] + p["handR"][1]), st.upper_arm, st.forearm, +1)
    knL, ftL = ik_outward(hipL, (hipL[0] + p["footL"][0], hipL[1] + p["footL"][1]), st.thigh, st.shin, -1)
    knR, ftR = ik_outward(hipR, (hipR[0] + p["footR"][0], hipR[1] + p["footR"][1]), st.thigh, st.shin, +1)
    toeL, toeR = (ftL[0] - st.foot, ftL[1]), (ftR[0] + st.foot, ftR[1])

    pts = dict(hip=hip, spine=spine, neck=neck, head=head, shL=shL, shR=shR, elL=elL, elR=elR, haL=haL, haR=haR,
               hipL=hipL, hipR=hipR, knL=knL, knR=knR, ftL=ftL, ftR=ftR, toeL=toeL, toeR=toeR)
    dy = floor_y - max(ftL[1], ftR[1])
    pts = {k: (x, y + dy) for k, (x, y) in pts.items()}
    pts["_head_angle"] = (ha, 0.0)
    return pts


# ------------------------------------------------------------- crack art
def _make_cracks() -> list[list[Vec]]:
    """Deterministic jagged polylines in head-local units (1.0 = head radius), radiating
    from the right temple across the head."""
    rng = np.random.default_rng(11)
    polys = []
    for ang, length in [(-200, 1.7), (-165, 1.35), (-232, 1.2), (-140, 0.95)]:
        a, pts, pos = math.radians(ang), [], np.array([0.88, -0.30])
        pts.append(tuple(pos))
        n = 7
        for _ in range(n):
            a += rng.uniform(-0.55, 0.55)
            pos = pos + np.array([math.cos(a), math.sin(a)]) * (length / n)
            pts.append(tuple(pos))
        polys.append(pts)
    return polys


CRACKS = _make_cracks()


def _partial(poly: list[Vec], t: float) -> list[Vec]:
    segs = [math.dist(a, b) for a, b in zip(poly, poly[1:])]
    target, out, acc = t * sum(segs), [poly[0]], 0.0
    for (a, b), L in zip(zip(poly, poly[1:]), segs):
        if acc + L <= target:
            out.append(b); acc += L
        else:
            r = (target - acc) / L
            out.append((a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r)); break
    return out


# ---------------------------------------------------------------- drawing
@dataclass
class Layers:
    x0: int
    y0: int
    outline: np.ndarray
    body: np.ndarray
    face_w: np.ndarray | None = None
    face_k: np.ndarray | None = None
    crack: np.ndarray | None = None


def _stroke(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], w: float, fill=255):
    d.line(pts, fill=fill, width=max(1, round(w)), joint="curve")
    r = w / 2
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def render_character(pose: dict, st: Style, region: tuple[int, int, int, int], floor_y: float,
                     silhouette_only: bool = False, crack: float = 0.0) -> Layers:
    """Draw one stickman. `region` is a box in final px; layers come back at world resolution."""
    S, SS = st.world_scale, st.supersample
    K = S * SS
    rx0, ry0, rx1, ry1 = region
    W, H = round((rx1 - rx0) * K), round((ry1 - ry0) * K)
    p = {**DEFAULT_POSE, **pose}
    pts = solve(p, st, floor_y)
    T = lambda q: ((q[0] - rx0) * K, (q[1] - ry0) * K)

    def figure(draw, extra: float, fill=255):
        e = extra * K
        w_body, w_limb = st.body_w * K + e, st.limb_w * K + e
        _stroke(draw, [T(pts["hip"]), T(pts["spine"])], w_body, fill)
        _stroke(draw, [T(pts["spine"]), T(pts["neck"])], st.limb_w * 0.8 * K + e, fill)
        _stroke(draw, [T(pts["shL"]), T(pts["shR"])], w_limb, fill)
        _stroke(draw, [T(pts["hipL"]), T(pts["hipR"])], w_limb, fill)
        for s in "LR":
            _stroke(draw, [T(pts["sh" + s]), T(pts["el" + s]), T(pts["ha" + s])], w_limb, fill)
            _stroke(draw, [T(pts["hip" + s]), T(pts["kn" + s]), T(pts["ft" + s]), T(pts["toe" + s])], w_limb, fill)
            hx, hy = T(pts["ha" + s]); r = st.hand_r * K + e / 2
            draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=fill)
        cx, cy = T(pts["head"]); r = st.head_r * K + e / 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    def down(img: Image.Image) -> np.ndarray:
        return np.asarray(img.resize((W // SS, H // SS), Image.BOX), np.float32) / 255.0

    body = Image.new("L", (W, H), 0)
    figure(ImageDraw.Draw(body), 0.0)
    out = Layers(round(rx0 * S), round(ry0 * S), None, down(body))
    ha = pts["_head_angle"][0]
    cx, cy = pts["head"]; r = st.head_r
    ca, sa = math.cos(ha), math.sin(ha)
    face = lambda u, v: T((cx + (u * ca - v * sa) * r, cy + (u * sa + v * ca) * r))   # head-local -> canvas

    def draw_cracks() -> np.ndarray | None:
        if crack <= 0:
            return None
        cr = Image.new("L", (W, H), 0)
        dc = ImageDraw.Draw(cr)
        for i, poly in enumerate(CRACKS):
            t = min(1.0, max(0.0, crack * 1.35 - i * 0.12))
            if t > 0:
                _stroke(dc, [face(u * 0.98, v * 0.98) for u, v in _partial(poly, t)], 0.08 * r * K)
        return down(cr)

    if silhouette_only:
        out.outline = out.body
        out.crack = draw_cracks()
        return out
    outline = Image.new("L", (W, H), 0)
    figure(ImageDraw.Draw(outline), 2 * st.outline)
    out.outline = down(outline)

    # ---- face ------------------------------------------------------------
    fw, fk = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    dw, dk = ImageDraw.Draw(fw), ImageDraw.Draw(fk)
    wide, blink = p["wide"], p["blink"]
    er = (0.20 + 0.07 * wide) * r * K
    for sgn in (-1, 1):
        ex, ey = face(sgn * 0.36, -0.06)
        h = er * max(0.08, 1 - blink)
        dw.ellipse([ex - er, ey - h, ex + er, ey + h], fill=255)
        if blink < 0.6:
            gx, gy = p["gaze"]
            px_, py_ = ex + gx * er * 0.45, ey + gy * er * 0.45
            pr = er * 0.46
            dk.ellipse([px_ - pr, py_ - pr * (1 - blink), px_ + pr, py_ + pr * (1 - blink)], fill=255)
        # eyebrows: `brow` = worry (inner ends up), `raise_` lifts both
        inner, outer = (sgn * 0.14, -0.40), (sgn * 0.60, -0.40)
        w_ = p["brow"]
        inner = (inner[0], inner[1] - 0.16 * w_ - 0.10 * p["raise_"])
        outer = (outer[0], outer[1] + 0.10 * w_ - 0.10 * p["raise_"])
        _stroke(dw, [face(*inner), face(*outer)], 0.075 * r * K)
    m = p["mouth"]
    mw = 0.07 * r * K
    if m == "flat":
        _stroke(dw, [face(-0.22, 0.46), face(0.22, 0.46)], mw)
    elif m == "o":
        mx, my = face(0, 0.46); mr = 0.12 * r * K
        dw.ellipse([mx - mr, my - mr * 1.15, mx + mr, my + mr * 1.15], fill=255)
        dk.ellipse([mx - mr * 0.55, my - mr * 0.65, mx + mr * 0.55, my + mr * 0.65], fill=255)
    elif m == "frown":
        _stroke(dw, [face(-0.24, 0.55), face(-0.10, 0.45), face(0.10, 0.45), face(0.24, 0.55)], mw)
    elif m == "wavy":
        _stroke(dw, [face(-0.26, 0.47), face(-0.13, 0.43), face(0.0, 0.49), face(0.13, 0.43), face(0.26, 0.47)], mw)
    elif m == "smile":
        _stroke(dw, [face(-0.24, 0.40), face(-0.10, 0.50), face(0.10, 0.50), face(0.24, 0.40)], mw)
    out.face_w, out.face_k = down(fw), down(fk)

    out.crack = draw_cracks()
    return out


def skeleton_bounds(pose: dict, st: Style, floor_y: float) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of the drawn figure in final px (outline included)."""
    pts = solve(pose, st, floor_y)
    xs = [v[0] for k, v in pts.items() if not k.startswith("_")]
    ys = [v[1] for k, v in pts.items() if not k.startswith("_")]
    pad = st.limb_w / 2 + st.outline
    hx, hy = pts["head"]
    return (min(min(xs) - pad, hx - st.head_r - st.outline), min(min(ys) - pad, hy - st.head_r - st.outline),
            max(max(xs) + pad, hx + st.head_r + st.outline), max(ys) + pad)
