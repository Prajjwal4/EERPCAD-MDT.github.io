"""Frame-level QA shared by every scene: format checks, caption-vs-audio sync, voice preview."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from caption_renderer import CaptionGroup, CaptionRenderer
from config import Config


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return {**j["streams"][0], "duration": float(j["format"]["duration"])}


def format_ok(p: dict) -> bool:
    return (p["codec_name"] == "h264" and p["pix_fmt"] == "yuv420p" and (p["width"], p["height"]) == (1080, 1920)
            and p["r_frame_rate"] == "30/1" and int(p["nb_read_frames"]) == 150 and abs(p["duration"] - 5.0) < 0.01)


def _frames(video: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf",
                          f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},format=rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, y1 - y0, x1 - x0, 3).astype(np.int32)


def verify_caption_sync(video: Path, events, cfg: Config, tolerance: int = 4) -> dict:
    """For every word of every kinetic caption event: find the first frame at which its ink is visible in the
    ENCODED video (inside that word's own bounding box) and compare with the audio-aligned start frame.
    Text appears after a short fade-in ramp, so the expected onset is start..start+tolerance."""
    cap = CaptionRenderer(cfg)
    band = (0, 1380, 1080, 1700)
    fr = _frames(video, band)
    rows, ok = [], True
    for ev in events:
        words = ev.group.words
        hit = cap.highlighted_indices(ev.group)
        for i, w in enumerate(words):
            ws = ev.word_starts[i]
            if ws <= -100:                                     # carried over from the previous scene
                rows.append((w, ws, None, "carried-in")); continue
            l, t, r, b = cap.build(ev.group, ev.kind, ev.position, only=frozenset([i])).ink_bbox
            reg = fr[:, t - band[1]:b - band[1], l:r]
            if i in hit:   # red word: red-dominant pixels
                ink = ((reg[..., 0] > 150) & (reg[..., 1] < 110) & (reg[..., 2] < 120)).sum(axis=(1, 2))
            else:          # white word
                ink = (reg.min(axis=3) > 120).sum(axis=(1, 2))
            thr = max(40, 0.08 * ink.max())
            # Another group (the outgoing one) may overlap this word's box until THIS group's start frame, so
            # only frames >= ev.start are attributed to this word. Ink inside the box between ev.start and the
            # word's aligned start would be a genuine early appearance.
            own = np.where(ink > thr)[0]
            own = own[own >= ev.start]
            onset = int(own[0]) if len(own) else None
            early = onset is not None and onset < ws
            good = onset is not None and (not early) and onset <= ws + tolerance
            ok &= good
            rows.append((w, ws, onset, "OK" if good else ("EARLY" if early else "CHECK")))
    return {"ok": ok, "rows": rows, "frames": len(fr),
           "empty_before_first_word": int(((fr.min(axis=3) > 120).sum(axis=(1, 2))[:min(e.start for e in events)]).max(initial=0))}


def persists_to_end(video: Path, last_event, cfg: Config) -> bool:
    """A group that continues into the next scene must still be fully on screen in the final frame."""
    cap = CaptionRenderer(cfg)
    ov = cap.build(last_event.group, last_event.kind, last_event.position)
    l, t, r, b = ov.ink_bbox
    fr = _frames(video, (l, t, r, b))
    return bool((fr[-1].min(axis=2) > 120).sum() > 0.5 * (fr[-2].min(axis=2) > 120).sum())


def voice_preview(video: Path, out: Path, cfg: Config, scene_number: int, spill_s: float) -> None:
    """QA clip: the scene + the matching slice of the real voiceover. If the line spills into the next scene, the final
    frame is held so the word finishes (this QA clip is longer than the 5 s scene; the scene file itself is not)."""
    t0 = (scene_number - 1) * 5.0
    dur = 5.0 + spill_s
    vf = f"tpad=stop_mode=clone:stop_duration={spill_s:.3f}" if spill_s > 0 else "null"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-ss", f"{t0}", "-t", f"{dur:.3f}",
                    "-i", str(cfg.path("assets/audio/voiceover_01.wav")), "-filter_complex", f"[0:v]{vf}[v]",
                    "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}", "-movflags", "+faststart", str(out)], check=True)


def report(p: dict, sync: dict, persists: bool | None = None) -> bool:
    print(f"format: {p['width']}x{p['height']} {p['r_frame_rate']} {p['pix_fmt']} frames={p['nb_read_frames']} duration={p['duration']:.3f}s ->",
          "OK" if format_ok(p) else "FAIL")
    print("caption sync (ink onset in the encoded video vs aligned word start):")
    for w, ws, on, st in sync["rows"]:
        print(f"   {w:<12} aligned f{ws:>4}   visible f{on if on is not None else '-':>4}   {st}")
    print("   nothing on screen before the first word (max ink in earlier frames):", sync["empty_before_first_word"])
    if persists is not None:
        print("   final group still on screen at the last frame (carries into next scene):", persists)
    return bool(format_ok(p) and sync["ok"] and (persists is None or persists))
