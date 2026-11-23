"""Caption / headline rendering: positioning, red highlighting, consistent shadow.

All text (captions and headlines) goes through `CaptionRenderer.build`, so every piece of
text gets the identical two-layer shadow. Overlays are cropped RGBA patches that are
composited onto the (already colour-graded) frame, so text is never tinted by the grade.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from config import Config

_STRIP = re.compile(r"[^\w'’-]+")


def _norm(word: str) -> str:
    return _STRIP.sub("", word).lower()


@dataclass(frozen=True)
class CaptionGroup:
    """One on-screen caption: <=7 words. `highlights` are terms (possibly multi-word) shown in red."""
    text: str
    highlights: tuple[str, ...] = ()

    @property
    def words(self) -> list[str]:
        return self.text.split()


@dataclass
class Overlay:
    x0: int
    y0: int
    rgb: np.ndarray      # planar float32 (3, h, w), 0..1, straight (non-premultiplied)
    alpha: np.ndarray    # float32 h x w, 0..1
    font_px: int
    lines: list[str]
    ink_bbox: tuple[int, int, int, int]  # absolute (l, t, r, b) of pixels with alpha above threshold


class CaptionRenderer:
    ALPHA_THRESHOLD = 8 / 255.0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.style = cfg.captions
        self.max_words = cfg.typography["max_words_per_group"]
        self.max_lines = cfg.typography["maximum_caption_lines"]
        self.sizes = {"caption": cfg.typography["caption_size_px"], "headline": cfg.typography["headline_size_px"]}
        self.colors = {"normal": cfg.rgb("primary_text"), "highlight": cfg.rgb(cfg.typography["highlight_color"])}
        self._fonts: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
        self._cache: dict[tuple, Overlay] = {}

    # ------------------------------------------------------------------ fonts
    def _font(self, kind: str, px: int) -> ImageFont.FreeTypeFont:
        key = (kind, px)
        if key not in self._fonts:
            rel = self.cfg.fonts["headline" if kind == "headline" else "body"]
            path = self.cfg.path(rel)
            try:
                self._fonts[key] = ImageFont.truetype(str(path), px)
            except OSError:
                fb = self.cfg.fonts.get("fallback")
                if not fb:
                    raise FileNotFoundError(f"Font {path} missing and no fallback configured") from None
                warnings.warn(f"Font {path} missing; using fallback {fb}", stacklevel=2)
                self._fonts[key] = ImageFont.truetype(str(self.cfg.path(fb)), px)
        return self._fonts[key]

    # -------------------------------------------------------------- highlight
    @staticmethod
    def highlighted_indices(group: CaptionGroup) -> set[int]:
        words = [_norm(w) for w in group.words]
        hit: set[int] = set()
        for term in group.highlights:
            toks = [_norm(t) for t in term.split()]
            n = len(toks)
            for i in range(len(words) - n + 1):
                if words[i:i + n] == toks:
                    hit.update(range(i, i + n))
        return hit

    # ----------------------------------------------------------------- layout
    @staticmethod
    def _wrap(words: list[str], font: ImageFont.FreeTypeFont, max_w: float) -> list[list[int]] | None:
        """Greedy wrap. Returns None if any single word is wider than max_w."""
        space = font.getlength(" ")
        lines: list[list[int]] = [[]]
        width = 0.0
        for i, w in enumerate(words):
            ww = font.getlength(w)
            if ww > max_w:
                return None
            add = ww if not lines[-1] else space + ww
            if lines[-1] and width + add > max_w:
                lines.append([i]); width = ww
            else:
                lines[-1].append(i); width += add
        return lines

    # ------------------------------------------------------------------ build
    def build(self, group: CaptionGroup, kind: str = "caption", position: str = "lower") -> Overlay:
        key = (group, kind, position)
        if key in self._cache:
            return self._cache[key]

        words = group.words
        if not words:
            raise ValueError("empty caption")
        if len(words) > self.max_words:
            raise ValueError(f"caption has {len(words)} words (max {self.max_words}): {group.text!r}")

        left, top, right, bottom = self.cfg.safe_rect
        pad = self.style["edge_pad_px"]
        max_w = (right - left) - 2 * pad
        max_h = (bottom - top) - 2 * pad
        spacing = self.style["line_spacing"]

        px = self.sizes[kind]
        while True:
            font = self._font(kind, px)
            wrapped = self._wrap(words, font, max_w)
            if wrapped is not None and len(wrapped) <= self.max_lines:
                asc, desc = font.getmetrics()
                lh = px * spacing
                block_h = asc + desc + (len(wrapped) - 1) * lh
                if block_h <= max_h:
                    break
            px -= 2
            if px < self.style["min_font_px"]:
                raise ValueError(f"cannot fit {group.text!r} in {self.max_lines} lines inside safe margins")

        # Canvas for this block (text + padding for the shadow).
        line_w = [sum(font.getlength(words[i]) for i in ln) + font.getlength(" ") * (len(ln) - 1) for ln in wrapped]
        cw, ch = int(np.ceil(max(line_w))) + 2 * pad, int(np.ceil(block_h)) + 2 * pad
        hit = self.highlighted_indices(group)

        def draw_text(draw: ImageDraw.ImageDraw, mono: int | None):
            y = pad + asc
            for ln, w in zip(wrapped, line_w):
                x = (cw - w) / 2
                for i in ln:
                    colour = mono if mono is not None else self.colors["highlight" if i in hit else "normal"]
                    draw.text((x, y), words[i], font=font, fill=colour, anchor="ls")
                    x += font.getlength(words[i]) + font.getlength(" ")
                y += lh

        # Text layer.
        text_img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        draw_text(ImageDraw.Draw(text_img), None)
        # Shadow: wide soft + tight crisp, from the same glyph mask.
        mask = Image.new("L", (cw, ch), 0)
        draw_text(ImageDraw.Draw(mask), 255)
        sh = self.style["shadow"]

        def shadow_alpha(blur, offset, opacity):
            m = Image.new("L", (cw, ch), 0)
            m.paste(mask, tuple(offset))
            return np.asarray(m.filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255.0 * opacity

        a1 = shadow_alpha(sh["blur_wide"], sh["offset_wide"], sh["opacity_wide"])
        a2 = shadow_alpha(sh["blur_tight"], sh["offset_tight"], sh["opacity_tight"])
        shadow_a = 1.0 - (1.0 - a1) * (1.0 - a2)
        shadow_rgb = np.asarray(self.cfg.rgb(sh["color"]), np.float32) / 255.0
        shadow = np.zeros((ch, cw, 4), np.float32)
        shadow[..., :3] = shadow_rgb
        shadow[..., 3] = shadow_a
        shadow_img = Image.fromarray((shadow * 255 + 0.5).astype(np.uint8), "RGBA")
        out = np.asarray(Image.alpha_composite(shadow_img, text_img), np.float32) / 255.0

        # Position inside the safe rectangle.
        x0 = (self.cfg.width - cw) // 2
        if position == "lower":
            y0 = bottom - ch
        elif position == "upper":
            y0 = top
        elif position == "center":
            y0 = (top + bottom - ch) // 2
        else:
            raise ValueError(f"unknown position {position!r}")

        alpha = out[..., 3]
        ys, xs = np.where(alpha > self.ALPHA_THRESHOLD)
        ink = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
        ov = Overlay(x0, y0, np.ascontiguousarray(out[..., :3].transpose(2, 0, 1)), alpha.copy(), px, [" ".join(words[i] for i in ln) for ln in wrapped], ink)
        self.check_safe(ov)
        self._cache[key] = ov
        return ov

    def check_safe(self, ov: Overlay) -> None:
        l, t, r, b = self.cfg.safe_rect
        il, it, ir, ib = ov.ink_bbox
        if il < l or it < t or ir > r or ib > b:
            raise ValueError(f"text outside safe margins: ink {ov.ink_bbox} vs safe {(l, t, r, b)}")

    # -------------------------------------------------------------- compositing
    def composite(self, frame: np.ndarray, ov: Overlay, opacity: float = 1.0) -> np.ndarray:
        """Blend overlay onto planar float32 (3,H,W) frame in place and return it."""
        if opacity <= 0:
            return frame
        h, w = ov.alpha.shape
        region = frame[:, ov.y0:ov.y0 + h, ov.x0:ov.x0 + w]
        a = ov.alpha * opacity
        region *= 1.0 - a
        region += ov.rgb * a
        return frame

    def event_opacity(self, local_i: int, length: int) -> float:
        n = self.style["fade_frames"]
        if n <= 0:
            return 1.0
        return float(min(1.0, (local_i + 1) / n, (length - local_i) / n))

    # ----------------------------------------------------------------- helpers
    def split_groups(self, text: str, highlights: tuple[str, ...] = ()) -> list[CaptionGroup]:
        """Split narration into groups of <= max_words, preferring punctuation breaks."""
        groups, cur = [], []
        for w in text.split():
            cur.append(w)
            if len(cur) >= self.max_words or (re.search(r"[.!?;:,]$", w) and len(cur) >= 3):
                groups.append(cur); cur = []
        if cur:
            if groups and len(cur) < 2 and len(groups[-1]) + len(cur) <= self.max_words:
                groups[-1] += cur
            else:
                groups.append(cur)
        return [CaptionGroup(" ".join(g), highlights) for g in groups]
