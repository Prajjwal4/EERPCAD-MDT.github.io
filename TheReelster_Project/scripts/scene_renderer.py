"""Scene renderer: cover-fit, camera zoom/pan, motion blur, colour grade, fades, captions.

Layer order per frame (planar float32 (3,H,W) until the final uint8 HWC conversion):
    image (cover-fit crop, motion-blurred) -> colour grade -> fade -> captions -> uint8
Captions go after the grade/fade so text is never tinted and stays readable across cuts.
Frames are piped to FFmpeg (libx264) - FFmpeg is the only encoder.
"""
from __future__ import annotations

import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from caption_renderer import CaptionGroup, CaptionRenderer, Overlay
from config import Config, load_config
from transition_engine import TransitionEngine

EASES = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) ** 2,
    "ease_in_out": lambda t: t * t * (3 - 2 * t),
}


# ----------------------------------------------------------------- data model
@dataclass
class Camera:
    """Zoom 1.0 = exact cover-fit. pan_* are -1..1 fractions of the slack left after zoom
    (x: -1 = left edge, +1 = right edge; y: -1 = top, +1 = bottom)."""
    zoom_start: float = 1.0
    zoom_end: float = 1.10
    pan_start: tuple[float, float] = (0.0, 0.0)
    pan_end: tuple[float, float] = (0.0, 0.0)
    ease: str = "ease_in_out"

    @property
    def is_static(self) -> bool:
        return self.zoom_start == self.zoom_end and self.pan_start == self.pan_end

    def at(self, t: float) -> tuple[float, tuple[float, float]]:
        e = EASES[self.ease](min(max(t, 0.0), 1.0))
        z = self.zoom_start + (self.zoom_end - self.zoom_start) * e
        pan = tuple(a + (b - a) * e for a, b in zip(self.pan_start, self.pan_end))
        return z, pan  # type: ignore[return-value]


@dataclass
class CaptionEvent:
    group: CaptionGroup
    start: int          # scene-local first frame (inclusive)
    end: int            # scene-local last frame (exclusive)
    kind: str = "caption"        # "caption" (66px) or "headline" (104px)
    position: str = "lower"      # lower | center | upper


@dataclass
class Scene:
    image: Path | None = None
    image_obj: Image.Image | None = None   # alternative to `image`, e.g. a temp test image
    camera: Camera | None = None           # None -> camera_defaults from config
    captions: list[CaptionEvent] = field(default_factory=list)
    frames: int | None = None              # None -> config scene_frames (150)
    fade_in: int | None = None
    fade_out: int | None = None


# ------------------------------------------------------------ colour grading
class ColorGrade:
    """Global look: desaturate, contrast, gamma, cool shadows / red-ish highlights, vignette,
    grain, and a black point lifted to the brand background colour.

    Works on PLANAR float32 frames (3, H, W): broadcasting a (H, W) plane over channels is
    several times faster in numpy than a trailing size-1 axis on (H, W, 3)."""

    GRAIN_TILES = 6
    _LUM = np.asarray([0.2126, 0.7152, 0.0722], np.float32)[:, None, None]

    def __init__(self, cfg: Config):
        g, (W, H) = cfg.grading, (cfg.width, cfg.height)
        self.sat, self.contrast, self.gamma, self.grain_sigma = g["saturation"], g["contrast"], g["gamma"], g["grain"]
        col = lambda name: np.asarray(cfg.rgb01(name), np.float32)[:, None, None]
        self.black = col(g["black_point"])
        self.sh_col, self.sh_str = col(g["shadow_tint"]["color"]), g["shadow_tint"]["strength"]
        self.hl_col, self.hl_str = col(g["highlight_tint"]["color"]), g["highlight_tint"]["strength"]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / np.sqrt(2)
        self.vignette = (1.0 - g["vignette"] * np.clip(r, 0, 1) ** 2.2).astype(np.float32)
        rng = np.random.default_rng(7)  # deterministic grain => reproducible renders
        self.grain = [rng.standard_normal((H, W), dtype=np.float32) * self.grain_sigma for _ in range(self.GRAIN_TILES)]

    BAND = 96  # rows per band; keeps temporaries in cache (the grade is memory-bandwidth bound)

    def apply(self, x: np.ndarray, frame_index: int) -> np.ndarray:
        """x: planar float32 (3,H,W) in 0..1. Returns a new array; input is not modified."""
        out = np.empty_like(x)
        gr = self.grain[frame_index % self.GRAIN_TILES]
        if (frame_index // self.GRAIN_TILES) % 2:
            gr = gr[::-1, ::-1]
        for r0 in range(0, x.shape[1], self.BAND):
            sl = slice(r0, r0 + self.BAND)
            out[:, sl] = self._band(x[:, sl], self.vignette[sl], gr[sl])
        return out

    def _band(self, x: np.ndarray, vig: np.ndarray, grain: np.ndarray) -> np.ndarray:
        w = self._LUM[:, 0, 0]
        lum = np.tensordot(w, x, axes=1)
        x = lum + (x - lum) * self.sat                                    # saturation
        np.clip(x, 0.0, 1.0, out=x)
        x -= 0.5; x *= self.contrast; x += 0.5                            # contrast
        np.clip(x, 0.0, 1.0, out=x)
        np.power(x, self.gamma, out=x)                                    # gamma (>1 = darker mids)
        lum = np.tensordot(w, x, axes=1)
        inv = 1.0 - lum
        x += (self.sh_col - x) * (inv * inv * self.sh_str)                # cool shadows
        x += (self.hl_col - x) * (lum * lum * lum * self.hl_str)          # red-ish highlights
        x *= vig
        x += grain
        x *= 1.0 - self.black
        x += self.black                                                   # black point = brand background
        return np.clip(x, 0.0, 1.0, out=x)


# --------------------------------------------------------------------- writer
class FFmpegWriter:
    """Pipes raw RGB frames into FFmpeg -> H.264 yuv420p (BT.709, faststart)."""

    def __init__(self, path: Path | str, cfg: Config, preset: str | None = None, crf: int | None = None):
        r = cfg.rendering
        self.cfg, self.count = cfg, 0
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "-",
               "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
               "-c:v", cfg.raw["project"]["video_codec"], "-preset", preset or r["final_preset"], "-crf", str(crf if crf is not None else r["crf"]),
               "-r", str(cfg.fps), "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-movflags", "+faststart", str(self.path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (self.cfg.height, self.cfg.width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"bad frame {frame.shape} {frame.dtype}")
        self.proc.stdin.write(frame.tobytes())
        self.count += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed:\n{err}")

    def __enter__(self): return self
    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.proc.kill()


# ------------------------------------------------------------------- renderer
class SceneRenderer:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.grade = ColorGrade(self.cfg)
        self.captions = CaptionRenderer(self.cfg)
        self.transitions = TransitionEngine(self.cfg)
        d = self.cfg.camera_defaults
        self.default_camera = Camera(d["zoom_start"], d["zoom_end"], ease=d["ease"])
        self.mb_samples, self.mb_shutter = self.cfg.motion_blur["samples"], self.cfg.motion_blur["shutter"]

    # -- source image ----------------------------------------------------
    def load_image(self, scene: Scene) -> Image.Image:
        if scene.image_obj is not None:
            img = scene.image_obj.convert("RGB")
        elif scene.image is not None:
            if not Path(scene.image).is_file():
                raise FileNotFoundError(f"visual asset missing: {scene.image}")  # never substitute
            img = Image.open(scene.image).convert("RGB")
        else:
            raise ValueError("scene needs image or image_obj")
        W, H = self.cfg.width, self.cfg.height
        if img.width < W or img.height < H:
            warnings.warn(f"source {img.size} smaller than canvas {W}x{H}; will be upscaled", stacklevel=2)
        # Cap resolution at ~2x canvas so zoom crops stay cheap but sharp.
        s = max(W / img.width, H / img.height)       # scale needed to cover
        if s < 0.5:
            img = img.resize((round(img.width * 2 * s), round(img.height * 2 * s)), Image.LANCZOS)
        return img

    def cover_box(self, img: Image.Image, zoom: float, pan: tuple[float, float]) -> tuple[float, float, float, float]:
        """Source-space crop with canvas aspect that fills the frame; zoom>=1 shrinks it."""
        sw, sh = img.size
        aspect = self.cfg.width / self.cfg.height
        w, h = (sh * aspect, sh) if sw / sh > aspect else (sw, sw / aspect)
        w, h = w / zoom, h / zoom
        px, py = (min(max(p, -1.0), 1.0) for p in pan)
        cx, cy = sw / 2 + px * (sw - w) / 2, sh / 2 + py * (sh - h) / 2
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def _sample(self, img: Image.Image, cam: Camera, t: float) -> np.ndarray:
        z, pan = cam.at(t)
        box = self.cover_box(img, z, pan)
        out = img.resize((self.cfg.width, self.cfg.height), Image.BICUBIC, box=box)
        return np.asarray(out, np.float32)  # HWC 0..255; scaled/planarised once per frame

    # -- frame -----------------------------------------------------------
    def visual_frame(self, img: Image.Image, cam: Camera, i: int, n: int) -> np.ndarray:
        """Motion blur = average of sub-frame samples spread over a fractional shutter."""
        denom = max(n - 1, 1)
        if cam.is_static or self.mb_samples <= 1:
            hwc = self._sample(img, cam, i / denom)
        else:
            offs = (np.arange(self.mb_samples) / (self.mb_samples - 1) - 0.5) * self.mb_shutter
            hwc = None
            for o in offs:
                s = self._sample(img, cam, (i + o) / denom)
                if hwc is None:
                    hwc = s
                else:
                    hwc += s
            hwc /= len(offs)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1)) * np.float32(1 / 255.0)   # planar (3,H,W) 0..1

    def iter_frames(self, scene: Scene, test_mode: bool = False) -> Iterator[np.ndarray]:
        n = scene.frames if scene.frames is not None else self.cfg.scene_frames
        if n != self.cfg.scene_frames and not test_mode:
            raise ValueError(f"scene must be exactly {self.cfg.scene_frames} frames, got {n}")
        img = self.load_image(scene)
        cam = scene.camera or self.default_camera
        overlays: list[tuple[CaptionEvent, Overlay]] = []
        for ev in scene.captions:
            if not (0 <= ev.start < ev.end <= n):
                raise ValueError(f"caption event {ev.start}-{ev.end} outside scene of {n} frames")
            overlays.append((ev, self.captions.build(ev.group, ev.kind, ev.position)))

        for i in range(n):
            x = self.visual_frame(img, cam, i, n)
            x = self.grade.apply(x, i)
            x = self.transitions.apply(x, i, n, scene.fade_in, scene.fade_out)
            for ev, ov in overlays:
                if ev.start <= i < ev.end:
                    self.captions.composite(x, ov, self.captions.event_opacity(i - ev.start, ev.end - ev.start))
            yield (np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8, order="C")

    def render_scene(self, scene: Scene, writer: FFmpegWriter, test_mode: bool = False) -> int:
        before = writer.count
        for frame in self.iter_frames(scene, test_mode):
            writer.write(frame)
        return writer.count - before
