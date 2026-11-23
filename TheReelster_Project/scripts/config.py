"""Single entry point for project_config.json.

Every other module gets its numbers from here; nothing is hard-coded elsewhere.
Invariants from the spec are checked at load time so a bad config fails loudly.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "project_config.json"


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


class ConfigError(ValueError):
    pass


class Config:
    def __init__(self, raw: dict, root: Path = ROOT):
        self.raw, self.root = raw, root
        p = raw["project"]
        self.width: int = p["resolution"]["width"]
        self.height: int = p["resolution"]["height"]
        self.fps: int = p["fps"]
        self.duration: float = p["duration_seconds"]
        self.total_frames: int = p["total_frames"]

        visuals = {k: v for k, v in raw["assets"].items() if k.startswith("visual_")}
        durations = {v["duration"] for v in visuals.values()}
        if len(durations) != 1:
            raise ConfigError(f"visual assets must all have the same duration, got {durations}")
        self.scene_count: int = len(visuals)
        self.scene_frames: int = int(round(durations.pop() * self.fps))

        # Spec invariants.
        if self.fps * self.duration != self.total_frames:
            raise ConfigError("fps * duration_seconds != total_frames")
        if self.scene_frames != 150:
            raise ConfigError(f"scenes must be exactly 150 frames, config gives {self.scene_frames}")
        if self.scene_frames * self.scene_count != self.total_frames:
            raise ConfigError("scene_frames * scene_count != total_frames")
        if (self.width, self.height) != (1080, 1920) or self.fps != 30:
            raise ConfigError("canvas must be 1080x1920 at 30 fps")

        self.identity: dict = raw["visual_identity"]
        self.typography: dict = raw["typography"]
        self.audio: dict = raw["audio"]
        self.rendering: dict = raw["rendering"]
        fw = raw["framework"]
        self.fonts, self.grading, self.camera_defaults = fw["fonts"], fw["grading"], fw["camera_defaults"]
        self.motion_blur, self.transitions = fw["motion_blur"], fw["transitions"]
        self.captions, self.audio_mix = fw["captions"], fw["audio_mix"]

    # -- colours ---------------------------------------------------------
    def rgb(self, name_or_hex: str) -> tuple[int, int, int]:
        """Colour by visual_identity key (e.g. 'accent_red') or literal '#RRGGBB'."""
        v = self.identity.get(name_or_hex, name_or_hex)
        return hex_rgb(v)

    def rgb01(self, name_or_hex: str) -> tuple[float, float, float]:
        return tuple(c / 255.0 for c in self.rgb(name_or_hex))  # type: ignore[return-value]

    # -- layout ----------------------------------------------------------
    @property
    def safe_rect(self) -> tuple[int, int, int, int]:
        """(left, top, right, bottom) in pixels; text incl. shadow must stay inside."""
        m = self.typography["safe_margins"]
        return m["left"], m["top"], self.width - m["right"], self.height - m["bottom"]

    def path(self, rel: str) -> Path:
        return self.root / rel


def load_config(path: Path | str = CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path) as f:
        return Config(json.load(f), root=path.parent)
