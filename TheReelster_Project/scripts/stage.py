"""Reusable dark-room stage for stickman scenes (same look as approved S01), with any number of red lamps.

Each lamp has its own pre-lit layer, so a scene can cross-fade between lights per frame (e.g. a wide beam
on a crowd -> a narrow spotlight on one character) for the cost of a few array multiplies.
All geometry is in FINAL 1080x1920 px; layers are stored at world resolution (x world_scale), planar (3,H,W).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import Config
from stickman import Style


def smooth_np(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fbm(h: int, w: int, octaves: int, seed: int, base: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc, amp, tot = np.zeros((h, w), np.float32), 1.0, 0.0
    for o in range(octaves):
        n = base * 2 ** o
        grid = rng.random((max(2, int(n * h / w) + 2), n + 2)).astype(np.float32)
        acc += amp * np.asarray(Image.fromarray(grid, "F").resize((w, h), Image.BICUBIC), np.float32)
        tot += amp; amp *= 0.5
    acc /= tot
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-6)


@dataclass
class Lamp:
    pos: tuple[float, float]       # bulb (final px)
    axis: tuple[float, float]      # point the beam aims at
    sigma: float = 0.46            # beam half-width (radians): small = focused spotlight
    gain: float = 1.9
    reach: float = 950.0           # distance falloff
    halo: float = 330.0            # soft glow radius around the bulb


class Stage:
    def __init__(self, cfg: Config, st: Style, lamps: list[Lamp], floor: float = 1340.0,
                 door: tuple[float, float, float] | None = (858, 455, 1012), seed: int = 3):
        S = st.world_scale
        self.S, self.floor, self.lamps = S, floor, lamps
        self.WW, self.WH = round(cfg.width * S), round(cfg.height * S)
        WW, WH = self.WW, self.WH
        self.FW, self.FH = cfg.width, cfg.height
        RED = np.asarray(cfg.rgb01("accent_red"), np.float32)
        BLUE = np.asarray(cfg.rgb01("research_blue"), np.float32)
        self.RED = RED
        X = (np.arange(WW, dtype=np.float32) / S)[None, :].repeat(WH, 0)
        Y = (np.arange(WH, dtype=np.float32) / S)[:, None].repeat(WW, 1)
        self.X, self.Y = X, Y

        # ---- structure lines (exact, Pillow)
        seams = Image.new("L", (WW, WH), 0); d = ImageDraw.Draw(seams)
        for x in range(0, cfg.width + 1, 180):
            d.line([(x * S, 0), (x * S, floor * S)], fill=255, width=3)
        d.line([(0, 1090 * S), (WW, 1090 * S)], fill=255, width=3)
        boards = Image.new("L", (WW, WH), 0); db = ImageDraw.Draw(boards)
        vp = (540 * S, (floor - 330) * S)
        for k in range(-9, 10):
            db.line([vp, (540 * S + k * 330 * S, WH)], fill=255, width=2)
        db.rectangle([0, 0, WW, floor * S], fill=0)
        trim = Image.new("L", (WW, WH), 0); dt = ImageDraw.Draw(trim)
        if door:
            dt.rectangle([door[0] * S, door[1] * S, door[2] * S, floor * S], outline=255, width=5)
        dt.line([(0, floor * S), (WW, floor * S)], fill=255, width=4)
        blur = lambda im: np.asarray(im.filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
        seams_a, boards_a, trim_a = blur(seams), blur(boards), blur(trim)
        tex = fbm(WH, WW, 5, seed)
        wall = 0.115 + 0.05 * (tex - 0.5)
        wall = np.where(Y > 1090, wall * 0.82, wall) * (1 - 0.55 * seams_a)
        if door:
            wall = np.where((X > door[0]) & (X < door[2]) & (Y > door[1]), wall * 0.42, wall)
        fl = (0.085 + 0.03 * (tex - 0.5)) * (1 - 0.4 * smooth_np((Y - floor) / 560)) * (1 - 0.6 * boards_a)
        self.alb = (np.where(Y < floor, wall, fl) + 0.05 * trim_a).astype(np.float32)
        self.dark = (self.alb[None] * (BLUE * 0.30)[:, None, None]).astype(np.float32)

        # ---- one pre-lit layer + beam + bulb sprite per lamp
        self.lit, self.beam, self.cone, self.bulb = [], [], [], []
        for lp in lamps:
            lx, ly = lp.pos
            dx, dy = X - lx, Y - ly
            dist = np.sqrt(dx * dx + dy * dy) + 1e-3
            th = np.arctan2(dy, dx) - math.atan2(lp.axis[1] - ly, lp.axis[0] - lx)
            th = (th + np.pi) % (2 * np.pi) - np.pi
            cone = np.exp(-((th / lp.sigma) ** 2)) / (1 + (dist / lp.reach) ** 2)
            halo = np.exp(-((dist / lp.halo) ** 2))
            gloss = np.exp(-((((X - (lp.axis[0] - 40)) / 300) ** 2) + (((Y - (floor + 130)) / 85) ** 2))) * (Y > floor)
            cone = np.clip(cone * lp.gain, 0, 1.5).astype(np.float32)
            I = cone + 0.55 * halo + 0.9 * gloss
            self.lit.append((self.alb[None] * RED[:, None, None] * (I[None] * 5.0)).astype(np.float32))
            self.beam.append((np.clip(cone, 0, 1) * 0.9 + 0.7 * halo).astype(np.float32))
            self.cone.append(cone)
            # bulb sprite on a small crop only
            R = 240
            x0, y0 = max(0, int((lx - R) * S)), max(0, int((ly - R) * S))
            x1, y1 = min(WW, int((lx + R) * S)), min(WH, int((ly + R) * S))
            ds = dist[y0:y1, x0:x1]
            sprite = (np.exp(-((ds / 11) ** 2))[None] * np.asarray([1.0, 0.78, 0.78], np.float32)[:, None, None] * 1.6
                      + (np.exp(-((ds / 75) ** 2)) * 0.5)[None] * RED[:, None, None]).astype(np.float32)
            self.bulb.append((x0, y0, sprite))
            cord = Image.new("L", (WW, WH), 0)
            ImageDraw.Draw(cord).line([(lx * S, 0), (lx * S, (ly - 14) * S)], fill=255, width=3)
            self.dark += (np.asarray(cord, np.float32) / 255 * 0.05)[None]
        self.haze_tex = fbm(WH + 220, WW + 420, 4, 9, base=4) ** 1.7

        # ---- dust
        rng = np.random.default_rng(5)
        n = 90
        self.px = rng.uniform(0, cfg.width, n); self.py = rng.uniform(150, 1400, n)
        self.pvx = rng.uniform(0.08, 0.45, n); self.pvy = rng.uniform(-0.45, -0.08, n)
        self.psize = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15]); self.pph = rng.uniform(0, 6.28, n)
        self.sprites = []
        for r in (1.4, 2.2, 3.4):
            yy, xx = np.mgrid[-6:7, -6:7].astype(np.float32)
            self.sprites.append(np.exp(-(xx * xx + yy * yy) / (2 * (r * S * 0.5) ** 2)).astype(np.float32))

    # ------------------------------------------------------------------ per frame
    def dust(self, f: float, ks: list[float]) -> np.ndarray:
        S, WW, WH = self.S, self.WW, self.WH
        P = np.zeros((WH, WW), np.float32)
        x = (self.px + self.pvx * f) % self.FW
        y = self.py + self.pvy * f
        for i in range(len(x)):
            wx, wy = int(x[i] * S), int(y[i] * S)
            if not (8 <= wx < WW - 8 and 8 <= wy < WH - 8):
                continue
            c = sum(k * float(cn[wy, wx]) for k, cn in zip(ks, self.cone))
            b = (0.10 * max(ks) + 1.1 * c) * (0.6 + 0.4 * math.sin(self.pph[i] + f * 0.2))
            P[wy - 6:wy + 7, wx - 6:wx + 7] += self.sprites[self.psize[i]] * b
        return P

    def compose(self, f: float, ks: list[float], unlit: np.ndarray) -> np.ndarray:
        """Dark room + every lamp (scaled by ks) + haze + bulbs + dust; `unlit` (H,W) = 1 where light is not blocked."""
        WW, WH, RED = self.WW, self.WH, self.RED
        frame = self.dark.copy()
        beam = np.zeros((WH, WW), np.float32)
        for lit, bm, k in zip(self.lit, self.beam, ks):
            if k > 0.003:
                frame += lit * (unlit * k)[None]
                beam += bm * k
        ox, oy = int(f * 1.6), int(f * 0.5)
        haze = self.haze_tex[oy:oy + WH, ox:ox + WW] * beam * unlit
        frame += (haze * 0.38)[None] * RED[:, None, None]
        for (x0, y0, sprite), k in zip(self.bulb, ks):
            if k > 0.003:
                frame[:, y0:y0 + sprite.shape[1], x0:x0 + sprite.shape[2]] += sprite * (k * (1 + 0.05 * math.sin(f * 1.3)))
        frame += self.dust(f, ks)[None] * (np.asarray([1.0, 0.62, 0.62], np.float32)[:, None, None] * 0.55 * max(ks))
        return frame
