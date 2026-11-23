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
    """RGB crops of every frame. The crop is aligned to even pixels (FFmpeg rounds odd sizes for yuv420) and cut back to `box`."""
    x0, y0, x1, y1 = box
    ex0, ey0, ex1, ey1 = x0 - (x0 % 2), y0 - (y0 % 2), x1 + (x1 % 2), y1 + (y1 % 2)
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf",
                          f"crop={ex1 - ex0}:{ey1 - ey0}:{ex0}:{ey0},format=rgb24", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, ey1 - ey0, ex1 - ex0, 3).astype(np.int32)
    return arr[:, y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]


def verify_caption_sync(video: Path, events, cfg: Config, tolerance: int = 4) -> dict:
    """For every word of every kinetic caption event: find the first frame at which its ink is visible in the
    ENCODED video (inside that word's own bounding box) and compare with the audio-aligned start frame.
    The pixel band that is decoded is derived from where the words really are (per caption position), so 3-line captions
    and upper headlines are covered. Text appears after a short fade-in ramp, so the expected onset is start..start+tolerance."""
    cap = CaptionRenderer(cfg)
    clean = video.parent / "clean" / f"{video.stem}_clean.mp4"
    clean = clean if clean.is_file() else None
    thr_px = 50 if clean is not None else 120              # difference threshold vs colour threshold
    boxes = {}                                            # (event index, word index) -> ink bbox
    for ei, ev in enumerate(events):
        for i in range(len(ev.group.words)):
            if ev.word_starts[i] > -100:
                boxes[(ei, i)] = cap.build(ev.group, ev.kind, ev.position, only=frozenset([i])).ink_bbox
    bands = {}
    for pos in {ev.position for ev in events}:
        bb = [boxes[(ei, i)] for (ei, i) in boxes if events[ei].position == pos]
        if bb:
            band = (0, max(0, min(b[1] for b in bb) - 8), 1080, min(1920, max(b[3] for b in bb) + 8))
            fr = _frames(video, band)
            if clean is not None:                          # caption ink = exact difference to the caption-free twin
                fr = np.abs(fr - _frames(clean, band)).max(axis=3, keepdims=True).repeat(3, axis=3)
            bands[pos] = (band, fr)
    n_frames = next(iter(bands.values()))[1].shape[0] if bands else 0
    rows, ok = [], True
    for ei, ev in enumerate(events):
        hit = cap.highlighted_indices(ev.group)
        for i, w in enumerate(ev.group.words):
            ws = ev.word_starts[i]
            if ws <= -100:                                 # carried over from the previous scene
                rows.append((w, ws, None, "carried-in")); continue
            if ws >= n_frames - 2:                         # begins at/after the last frame: appears in the next scene
                rows.append((w, ws, None, "next-scene")); continue
            l, t, r, b = boxes[(ei, i)]
            band, fr = bands[ev.position]
            reg = fr[:, t - band[1]:b - band[1], l:r]
            if clean is not None:
                ink = (reg[..., 0] > thr_px).sum(axis=(1, 2))
            elif i in hit:   # fallback without the caption-free twin: red-dominant pixels
                ink = ((reg[..., 0] > 150) & (reg[..., 1] < 110) & (reg[..., 2] < 120)).sum(axis=(1, 2))
            else:            # white word
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
    first_start = min((e.start for e in events), default=0)
    stray = 0
    for band, fr in bands.values():
        stray = max(stray, int((fr.min(axis=3) > thr_px).sum(axis=(1, 2))[:first_start].max(initial=0)))
    return {"ok": ok, "rows": rows, "frames": n_frames, "empty_before_first_word": stray}


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
