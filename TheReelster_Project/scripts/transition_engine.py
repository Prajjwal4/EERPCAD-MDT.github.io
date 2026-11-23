"""Fade transitions.

The timeline is a fixed budget (12 scenes x 150 frames = 1800), so transitions never
overlap scenes: each scene fades up from, and back down to, the brand background colour
*inside* its own 150 frames. A true cross-dissolve between scenes would consume frames
from both sides and break the 1800-frame total, so `crossfade` exists only as a
primitive for effects within a scene.
"""
from __future__ import annotations

import numpy as np

from config import Config


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade_factor(i: int, total: int, fade_in: int, fade_out: int) -> float:
    """Visibility 0..1 for frame i of `total`. Frame 0 is fully faded; the last frame too."""
    f = 1.0
    if fade_in > 0 and i < fade_in:
        f = min(f, _smooth(i / fade_in))
    if fade_out > 0 and i > total - 1 - fade_out:
        f = min(f, _smooth((total - 1 - i) / fade_out))
    return f


def apply_fade(frame: np.ndarray, factor: float, color01: tuple[float, float, float]) -> np.ndarray:
    """Blend a planar float32 (3,H,W) frame towards `color01` (factor 1 = untouched)."""
    if factor >= 1.0:
        return frame
    c = np.asarray(color01, np.float32)[:, None, None]
    return frame * factor + c * (1.0 - factor)


def crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear blend of two float frames, t=0 -> a, t=1 -> b."""
    return a * (1.0 - t) + b * t


class TransitionEngine:
    def __init__(self, cfg: Config):
        t = cfg.transitions
        self.fade_in, self.fade_out = t["fade_in_frames"], t["fade_out_frames"]
        self.color = cfg.rgb01(t["fade_color"])

    def apply(self, frame: np.ndarray, i: int, total: int,
              fade_in: int | None = None, fade_out: int | None = None) -> np.ndarray:
        fi = self.fade_in if fade_in is None else fade_in
        fo = self.fade_out if fade_out is None else fade_out
        return apply_fade(frame, fade_factor(i, total, fi, fo), self.color)
