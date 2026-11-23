"""Props toolkit for stickman scenes. Everything is specified in FINAL 1080x1920 px; frames are planar float32 (3,H,W) at
world resolution. Shapes are drawn with Pillow (2x supersampled), then composited ('over') or added ('add') with optional glow.

    paint(frame, S, bbox, draw_fn, color, alpha=1, mode="over", glow=0, glow_gain=0)
    glow_disc(frame, S, cx, cy, r, color, k)
    text_mask / paint_text(frame, S, cfg, "ONLY 3 LEFT", cx, cy, px, color, alpha, glow)
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2


def _box(frame, S, bbox):
    H, W = frame.shape[1], frame.shape[2]
    x0, y0, x1, y1 = bbox
    X0, Y0, X1, Y1 = int(x0 * S), int(y0 * S), int(np.ceil(x1 * S)), int(np.ceil(y1 * S))
    cx0, cy0, cx1, cy1 = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return X0, Y0, cx0, cy0, cx1, cy1


def mask_from(frame, S, bbox, draw_fn):
    """Render draw_fn into an 'L' mask for bbox (final px). draw_fn(draw, T) where T maps final px -> canvas px.
    Returns (frame slices, float mask 0..1 of the visible part)."""
    b = _box(frame, S, bbox)
    if b is None:
        return None
    X0, Y0, cx0, cy0, cx1, cy1 = b
    w, h = cx1 - cx0, cy1 - cy0
    img = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(img)
    k = S * SS
    T = lambda x, y: ((x * S - cx0) * SS, (y * S - cy0) * SS)
    Tl = k  # scalar: final px length -> canvas px length
    draw_fn(d, T, Tl)
    m = np.asarray(img.resize((w, h), Image.BOX), np.float32) / 255.0
    return (slice(cy0, cy1), slice(cx0, cx1)), m


def paint(frame, S, bbox, draw_fn, color, alpha=1.0, mode="over", glow=0.0, glow_gain=0.0):
    r = mask_from(frame, S, bbox, draw_fn)
    if r is None:
        return
    (sy, sx), m = r
    col = np.asarray(color, np.float32)[:, None, None]
    reg = frame[:, sy, sx]
    if mode == "over":
        a = (m * alpha)[None]
        reg[:] = reg * (1 - a) + col * a
    else:
        reg += col * (m * alpha)[None]
    if glow > 0 and glow_gain > 0:
        g = np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(glow * S)), np.float32) / 255.0
        reg += col * (g * glow_gain * alpha)[None]


def glow_disc(frame, S, cx, cy, r, color, k=1.0, falloff=1.0):
    R = r * 3.2
    b = _box(frame, S, (cx - R, cy - R, cx + R, cy + R))
    if b is None or k <= 0:
        return
    X0, Y0, x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = ((xx / S - cx) ** 2 + (yy / S - cy) ** 2) / (r * r)
    frame[:, y0:y1, x0:x1] += (np.exp(-d2 * falloff) * k)[None] * np.asarray(color, np.float32)[:, None, None]


def font(cfg, px: float, S: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(cfg.path(cfg.fonts["headline"])), max(4, int(round(px * S * SS))))


def paint_text(frame, S, cfg, text, cx, cy, px, color, alpha=1.0, mode="over", glow=0.0, glow_gain=0.0, spacing=0.0):
    f = font(cfg, px, S)
    wpx = f.getlength(text) / (S * SS) + spacing * len(text)
    bbox = (cx - wpx / 2 - 30, cy - px, cx + wpx / 2 + 30, cy + px)

    def draw(d, T, Tl):
        x = T(cx - wpx / 2, cy)[0]
        if spacing:
            for ch in text:
                d.text((x, T(0, cy)[1]), ch, font=f, fill=255, anchor="lm")
                x += f.getlength(ch) + spacing * Tl
        else:
            d.text((T(cx, cy)[0], T(cx, cy)[1]), text, font=f, fill=255, anchor="mm")
    paint(frame, S, bbox, draw, color, alpha, mode, glow, glow_gain)


def paint_line(frame, S, pts, width, color, alpha=1.0, mode="over", glow=0.0, glow_gain=0.0):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = width + glow * 3 + 4
    bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def draw(d, T, Tl):
        q = [T(*p) for p in pts]
        d.line(q, fill=255, width=max(1, int(width * Tl)), joint="curve")
        r = width * Tl / 2
        for x, y in (q[0], q[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    paint(frame, S, bbox, draw, color, alpha, mode, glow, glow_gain)


def paint_rect(frame, S, x0, y0, x1, y1, color, alpha=1.0, mode="over", radius=0.0, outline=0.0, glow=0.0, glow_gain=0.0):
    pad = glow * 3 + 6
    bbox = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)

    def draw(d, T, Tl):
        a, b = T(x0, y0), T(x1, y1)
        if outline:
            d.rounded_rectangle([a, b], radius=radius * Tl, outline=255, width=max(1, int(outline * Tl)))
        else:
            d.rounded_rectangle([a, b], radius=radius * Tl, fill=255)
    paint(frame, S, bbox, draw, color, alpha, mode, glow, glow_gain)


def paint_poly(frame, S, pts, color, alpha=1.0, mode="over", glow=0.0, glow_gain=0.0):
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = glow * 3 + 6
    bbox = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
    paint(frame, S, bbox, lambda d, T, Tl: d.polygon([T(*p) for p in pts], fill=255), color, alpha, mode, glow, glow_gain)


def paint_ellipse(frame, S, cx, cy, rx, ry, color, alpha=1.0, mode="over", outline=0.0, glow=0.0, glow_gain=0.0):
    pad = glow * 3 + 6
    bbox = (cx - rx - pad, cy - ry - pad, cx + rx + pad, cy + ry + pad)

    def draw(d, T, Tl):
        a, b = T(cx - rx, cy - ry), T(cx + rx, cy + ry)
        if outline:
            d.ellipse([a, b], outline=255, width=max(1, int(outline * Tl)))
        else:
            d.ellipse([a, b], fill=255)
    paint(frame, S, bbox, draw, color, alpha, mode, glow, glow_gain)
