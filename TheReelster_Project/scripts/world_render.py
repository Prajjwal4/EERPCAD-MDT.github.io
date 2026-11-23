"""Shared compositor for multi-character stickman scenes on a Stage (generalised from approved S02).

render(stage, style, cfg, f, actors, ks, fx=None):
    actors : [(name, pose, floor_y, scale)]  - drawn far-to-near
    ks     : per-lamp intensities (one per stage lamp)
    fx     : optional callback fx(frame, f, acts, ks) for scene-specific effects drawn on top (ripples, tears, ...)
Cast shadows: floor shadow for everyone (away from the dominant lamp), big wall shadow for the character named "P".
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from config import Config
from stage import Stage
from stickman import Style, render_character


def _affine(img: Image.Image, coef) -> np.ndarray:
    return np.asarray(img.transform(img.size, Image.AFFINE, coef, Image.BILINEAR), np.float32) / 255


def _blur(a: np.ndarray, r: float) -> np.ndarray:
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255


def clip_view(WW, WH, x0: int, y0: int, h: int, w: int):
    fx0, fy0, fx1, fy1 = max(x0, 0), max(y0, 0), min(x0 + w, WW), min(y0 + h, WH)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    lx, ly = fx0 - x0, fy0 - y0
    return (slice(fy0, fy1), slice(fx0, fx1)), (slice(ly, ly + fy1 - fy0), slice(lx, lx + fx1 - fx0))


def render(stage: Stage, st: Style, cfg: Config, f: float, actors, ks: list[float], fx=None, wall_dx_limits=(170, 230)):
    S, WW, WH = st.world_scale, stage.WW, stage.WH
    RED = stage.RED
    GREY = np.asarray(cfg.rgb01("secondary_text"), np.float32)
    WHITE = np.asarray(cfg.rgb01("primary_text"), np.float32)
    kw, ksolo = ks[0], ks[1]
    lamp_x = sum(k * lp.pos[0] for k, lp in zip(ks, stage.lamps)) / max(sum(ks), 1e-3)

    acts = []
    for name, pose, fl, sc in sorted(actors, key=lambda a: a[2]):
        sty = st.scaled(sc)
        hx = pose["hip_x"]
        region = (int(hx - 330 * sc), int(fl - 860 * sc), int(hx + 330 * sc), int(fl + 120))
        acts.append((name, pose, fl, sc, hx, render_character(pose, sty, region, fl)))

    floor_sh = np.zeros((WH, WW), np.float32)
    wall_sh = np.zeros((WH, WW), np.float32)
    for name, pose, fl, sc, hx, L in acts:
        h, w = L.body.shape
        FLW = fl * S
        b = float(np.clip((hx - lamp_x) / 700.0, -0.9, 0.9))
        FA = 0.28
        coef = (1, -b / FA, (b / FA) * FLW, 0, -1 / FA, FLW * (1 + 1 / FA))
        if clip_view(WW, WH, L.x0, L.y0, h, w) is None:
            continue
        full = Image.new("L", (WW, WH), 0)
        full.paste(Image.fromarray((np.clip(L.body, 0, 1) * 255).astype(np.uint8), "L"), (L.x0, L.y0))
        floor_sh = np.maximum(floor_sh, _affine(full, coef) * 0.9)
        if name == "P":
            sgn = 1.0 if hx >= lamp_x else -1.0
            m, dx = 1.28, sgn * float(np.clip(abs(hx - lamp_x) * 0.40, *wall_dx_limits))
            axw, axw2 = hx * S, (hx + dx) * S
            wall_sh = _affine(full, (1 / m, 0, axw - axw2 / m, 0, 1 / m, FLW - FLW / m))
    shadow = np.maximum(_blur(wall_sh, 5), _blur(floor_sh, 4))
    frame = stage.compose(f, ks, 1 - 0.93 * shadow)

    for name, pose, fl, sc, hx, L in acts:
        h, w = L.body.shape
        v = clip_view(WW, WH, L.x0, L.y0, h, w)
        if v is None:
            continue
        (fy, fx_), (ly, lx) = v
        reg = frame[:, fy, fx_]
        ol, bd = L.outline[ly, lx], L.body[ly, lx]
        vis = 0.30 + 0.70 * min(1.0, (kw if name != "P" else max(kw * 0.5, ksolo)))
        reg[:] = reg * (1 - ol[None]) + (GREY * 0.42 * vis)[:, None, None] * ol[None]
        reg *= (1 - bd[None])
        reg += bd[None] * 0.012
        if name == "P":   # lit from above by the spotlight (+ a little from the crowd's beam)
            rim = np.clip(ol - np.roll(ol, 9, axis=0), 0, 1) * 1.25 * ksolo + np.clip(ol - np.roll(ol, -9, axis=1), 0, 1) * 0.8 * kw
        else:
            rim = np.clip(ol - np.roll(ol, -9, axis=1), 0, 1) * 1.35 * kw
        reg += (_blur(rim, 1.2))[None] * RED[:, None, None]
        fr_row = round(fl * S) - L.y0
        d = min(h - fr_row, 110)
        if d > 0 and fr_row - d > 0:
            refl = ol[fr_row - d:fr_row][::-1]
            grad = np.linspace(0.45, 0.0, d, dtype=np.float32)[:, None]
            reg[:, fr_row:fr_row + d] += (refl * grad)[None] * (RED * 0.5 + GREY * 0.12)[:, None, None] * (ksolo if name == "P" else kw)
        if L.face_w is not None:
            fw, fk = L.face_w[ly, lx], L.face_k[ly, lx]
            reg[:] = reg * (1 - 0.96 * fw[None]) + (WHITE * (0.45 + 0.55 * vis))[:, None, None] * (0.96 * fw[None])
            reg *= (1 - 0.98 * fk[None])
    if fx is not None:
        fx(frame, f, acts, ks)
    return frame
