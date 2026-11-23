"""Shared scene runner: build -> render (captions burned in + caption-free copy) -> QA -> preview / contact / keyframe.

A scene module provides: world_image(f), build_scene(), layout_check() (optional) and calls run(...).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import qa_scene
import world_render
from config import load_config
from scene_renderer import SceneRenderer
from stage import Stage

cfg = load_config()


def make_world_image(stage: Stage, st, actors_fn, lights_fn, pre_fx=None, fx=None, offset: int = 0):
    """Returns world_image(f) -> PIL image. `offset` is added to the time handed to the stage (haze/dust continuity)."""
    def world_image(f: float) -> Image.Image:
        fr = world_render.render(stage, st, cfg, f + offset, actors_fn(f), lights_fn(f),
                                 fx=(lambda frame, F, acts, ks: fx(frame, F - offset, acts, ks)) if fx else None,
                                 pre_fx=(lambda frame, F, ks: pre_fx(frame, F - offset, ks)) if pre_fx else None)
        return Image.fromarray((np.clip(fr, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8), "RGB")
    return world_image


def to_png(img: Image.Image, R: SceneRenderer, f: int, path: Path) -> None:
    x = R._planar(np.asarray(img.resize((cfg.width, cfg.height), Image.LANCZOS), np.float32))
    x = R.grade.apply(x, f)
    Image.fromarray((np.clip(x, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(path)


def stills(world_image, R, frames, prefix) -> None:
    for f in frames:
        to_png(world_image(f), R, f, cfg.path(f"qa/{prefix}_dev_{f:03d}.png"))
    ims = [Image.open(cfg.path(f"qa/{prefix}_dev_{f:03d}.png")).resize((360, 640)) for f in frames]
    sheet = Image.new("RGB", (360 * len(ims), 640))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * 360, 0))
    sheet.save(cfg.path(f"qa/{prefix}_dev_sheet.png"))


def run(n: int, build_scene, world_image, R: SceneRenderer, preview_f: int, key_f: int, contact_fs, layout_fn=None) -> bool:
    sid = f"S{n:02d}"
    if layout_fn:
        lay = layout_fn()
        print("layout:", lay)
        if not lay.get("ok", True):
            sys.exit(f"{sid}: layout check failed")
    scene = build_scene()
    out = cfg.path(f"scenes/{sid}.mp4")
    clean = cfg.path(f"scenes/clean/{sid}_clean.mp4")
    qa_only = bool(__import__('os').environ.get("REELSTER_QA_ONLY"))
    cnt = 150 if qa_only else R.render_scene_files(scene, out, clean)
    p = qa_scene.probe(out)
    pc = qa_scene.probe(clean)
    sync = qa_scene.verify_caption_sync(out, scene.captions, cfg) if any(e.word_starts for e in scene.captions) else {"ok": True, "rows": [], "empty_before_first_word": 0}
    persists = qa_scene.persists_to_end(out, scene.captions[-1], cfg) if scene.captions and not scene.captions[-1].fade_out else None
    ok = qa_scene.report(p, sync, persists) and cnt == 150 and qa_scene.format_ok(pc)
    print("caption-free copy:", "OK" if qa_scene.format_ok(pc) else "FAIL")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf", f"select=eq(n\\,{preview_f})",
                    "-frames:v", "1", str(cfg.path(f"qa/{sid}_preview.png"))], check=True)
    sel = "+".join(f"eq(n\\,{f})" for f in contact_fs)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), "-vf",
                    f"select='{sel}',scale=270:480,tile={len(contact_fs)}x1", "-frames:v", "1", str(cfg.path(f"qa/{sid}_contact.png"))], check=True)
    to_png(world_image(key_f), R, 0, cfg.path(f"assets/visuals/visual_{n:02d}.png"))
    print(f"{sid} VERIFIED" if ok else f"{sid} FAILED VERIFICATION")
    return ok


def hand_target(pose: dict, side: str, target, st, floor: float):
    """Hand target relative to the shoulder (what the rig wants) so that the hand reaches the absolute point `target` (final px)."""
    from stickman import solve
    base = solve({**pose, "handR": (34.0, 232.0), "handL": (-34.0, 232.0)}, st, floor)
    sh = base["shR" if side == "R" else "shL"]
    return (target[0] - sh[0], target[1] - sh[1])
