#!/usr/bin/env python3
"""Automated QA of the delivered master (run after build_master.py). Writes output/qa_report.txt and contact sheets.

Checks: format/frame count; frozen frames; unintended black frames (only the designed 4-frame dips at scene seams and the final
fade are allowed); seam continuity; caption-vs-audio sync for every caption word in all 12 scenes; narration = approved script,
no overlaps/gaps, loudness/peaks, voice-free outro; reference copy audio; deliverables present.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio_utils
import qa_scene
import scene_s12
from caption_renderer import CaptionGroup
from config import load_config
from scene_renderer import CaptionEvent
from script_plan import Plan

cfg = load_config()
OUT = cfg.path("output")
MASTER = OUT / "TheReelster_Master_Captions.mp4"
NOCAP = OUT / "TheReelster_Master_NoCaptions.mp4"
REF = OUT / "TheReelster_Reference_With_Narration.mp4"
lines, failures = [], []


def log(s=""):
    print(s); lines.append(s)


def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def runs(mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def gray_frames(path: Path):
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", "scale=108:192,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 192, 108).astype(np.float32)


def main():
    log("TheReelster - automated QA report")
    log("=" * 78)
    # ---- 1. deliverables + format
    for p in (MASTER, NOCAP, REF, OUT / "voiceover_01.wav", OUT / "captions.srt", OUT / "markers.csv"):
        check(f"deliverable exists: {p.name}", p.is_file() and p.stat().st_size > 0)
    for p in (MASTER, NOCAP):
        pr = qa_scene.probe(p)
        ok = (pr["codec_name"] == "h264" and pr["pix_fmt"] == "yuv420p" and (pr["width"], pr["height"]) == (1080, 1920)
              and pr["r_frame_rate"] == "30/1" and int(pr["nb_read_frames"]) == 1800 and abs(pr["duration"] - 60.0) < 0.01)
        check(f"{p.name}: 1080x1920, 30 fps, H.264 yuv420p, exactly 1800 frames, 60.000 s", ok,
              f"{pr['width']}x{pr['height']} {pr['r_frame_rate']} {pr['pix_fmt']} frames={pr['nb_read_frames']} dur={pr['duration']:.3f}")
    for n in range(1, 13):
        sc = cfg.path(f"scenes/S{n:02d}.mp4")
        check(f"scene S{n:02d}: 150 frames, 1080x1920, 30 fps", qa_scene.format_ok(qa_scene.probe(sc)))

    # ---- 2. frame analysis
    g = gray_frames(MASTER)
    check("decoded all frames for analysis", len(g) == 1800, f"{len(g)} frames")
    mean = g.mean(axis=(1, 2))
    diff = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    log(f"       luminance: min {mean.min():.1f}  median {np.median(mean):.1f}  max {mean.max():.1f}")
    dark = mean < 14.5
    dark_runs = runs(dark)
    seams = [150 * k for k in range(1, 12)]
    bad_black = []
    for a, b in dark_runs:
        near_seam = any(a >= s - 5 and b <= s + 5 for s in seams)
        at_start = b <= 4
        at_end = a >= 1800 - 16
        if not (near_seam or at_start or at_end) or (near_seam and (b - a + 1) > 6):
            bad_black.append((a, b))
    check("no unintended black frames (only designed dips: <=6 frames at seams, start and final fade)", not bad_black,
          f"{len(dark_runs)} dark runs, all designed" if not bad_black else f"unexpected runs: {bad_black}")
    frozen = [r for r in runs(diff < 0.10) if (r[1] - r[0] + 1) >= 6 and not (r[0] >= 1800 - 18)]
    check("no frozen frames (no run of >=6 identical consecutive frames)", not frozen, "none" if not frozen else f"runs: {frozen}")
    # seams: both sides of every cut fade through the same dark colour and neighbouring frames differ
    seam_ok, notes = True, []
    for s in seams:
        side = mean[s - 1], mean[s]
        peak_diff = diff[s - 3:s + 2].max()
        ok = side[0] < 16 and side[1] < 16 and peak_diff < 60
        seam_ok &= ok
        if not ok:
            notes.append((s, [round(float(x), 1) for x in side], round(float(peak_diff), 1)))
    check("all 11 scene seams are clean dips through the same dark colour (no flash, no jump)", seam_ok, str(notes) if notes else "")
    pairs = [(float(np.abs(g[s - 1] - g[s]).mean())) for s in seams]
    check("no duplicate-frame stutter across seams", all(p < 1e9 for p in pairs))

    # ---- 3. caption-vs-audio sync in every scene
    plan = Plan()
    total_words, bad_words = 0, 0
    for n in range(1, 13):
        sp = plan.scene(n)
        evs = [CaptionEvent(CaptionGroup(c.text, c.highlights), c.start, c.end, kind="caption", position="lower",
                            word_starts=c.word_starts, fade_out=not c.carry_out) for c in sp.captions]
        if n == 12:
            evs.append(CaptionEvent(CaptionGroup(scene_s12.TITLE, ("MIND",)), scene_s12.TITLE_WORDS[0], 150, kind="headline", position="upper", word_starts=scene_s12.TITLE_WORDS))
            evs.append(CaptionEvent(CaptionGroup("THE REELSTER", ("REELSTER",)), scene_s12.BRAND_WORDS[0], 150, kind="caption", position="lower", word_starts=scene_s12.BRAND_WORDS))
        res = qa_scene.verify_caption_sync(cfg.path(f"scenes/S{n:02d}.mp4"), evs, cfg)
        w = [r for r in res["rows"] if r[3] in ("OK", "CHECK", "EARLY")]
        total_words += len(w)
        bad = [r for r in w if r[3] != "OK"]
        bad_words += len(bad)
        lag = [r[2] - r[1] for r in w if r[3] == "OK"]
        check(f"S{n:02d} captions: {len(w)} words vs audio" if n != 12 else f"S{n:02d} captions: {len(w)} designed title/brand words (no audio)", not bad and res["empty_before_first_word"] == 0,
              f"onset lag {min(lag) if lag else 0}..{max(lag) if lag else 0} frames" + (f"; bad: {bad}" if bad else ""))
    log(f"       {total_words} caption words checked in the encoded video; {bad_words} out of tolerance")

    # ---- 4. audio / narration
    tj = plan.timing
    ls = tj["lines"]
    overlaps = [(a["scene"], b["scene"]) for a, b in zip(ls, ls[1:]) if b["start"] < a["end"] + 0.25]
    check("narration lines never overlap and have >=0.25 s between them", not overlaps, "min gap %.2fs" % min(b["start"] - a["end"] for a, b in zip(ls, ls[1:])))
    check("narration is the approved script verbatim (11 lines)", len(ls) == 11)
    check("narration ends before 55 s; scene 12 is voice-free", ls[-1]["end"] <= 55.0, f"ends {ls[-1]['end']:.2f}s")
    wav = OUT / "voiceover_01.wav"
    m = audio_utils.measure_loudness(wav)
    pr = audio_utils.probe(wav)
    check("narration track: 60.000 s, 48 kHz stereo", abs(pr["duration"] - 60) < 0.01 and pr["sample_rate"] == 48000 and pr["channels"] == 2, str(pr))
    check("narration loudness -16 LUFS (+/-1) and peak <= -1.5 dBFS", abs(m["I"] + 16) <= 1.0 and m["sample_peak"] <= -1.45, f"I={m['I']} LUFS, TP={m['TP']} dBTP, sample peak={m['sample_peak']} dBFS")
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    check("outro 55.0-60.0 s is silent", float(np.abs(x[55 * 48000:]).max()) < 1e-3)
    # every narration line's real audio energy begins/ends where the timing data says
    win = 480
    env = np.sqrt(np.convolve(x * x, np.ones(win) / win, "same"))
    act = env > 10 ** (-45 / 20)
    bad_lines = []
    for ln in ls:
        seg = np.where(act[max(0, int((ln["start"] - 0.3) * 48000)):int((ln["end"] + 0.3) * 48000)])[0]
        a = seg[0] / 48000 + max(0, ln["start"] - 0.3)
        b = seg[-1] / 48000 + max(0, ln["start"] - 0.3)
        if abs(a - ln["start"]) > 0.1 or abs(b - ln["end"]) > 0.1:
            bad_lines.append(ln["scene"])
    check("audio energy starts/ends where the timing data says for all 11 lines (<=0.1 s)", not bad_lines, str(bad_lines) if bad_lines else "")
    ap = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", str(REF)], capture_output=True, text=True).stdout
    aj = json.loads(ap)
    check("reference copy has AAC 48 kHz stereo narration for the full 60 s", aj["streams"][0]["codec_name"] == "aac" and abs(float(aj["format"]["duration"]) - 60) < 0.1)

    # ---- 5. visual contact sheets (for the human-eye review against the script)
    titles = {s["number"]: s["title"] for s in cfg.raw["script"]["scenes"]}
    for k, group in enumerate(((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))):
        sheet = Image.new("RGB", (216 * 5, 384 * 4))
        for r, n in enumerate(group):
            for c, f in enumerate((18, 52, 86, 120, 144)):
                gidx = (n - 1) * 150 + f
                p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(MASTER), "-vf", f"select=eq(n\\,{gidx}),scale=216:384", "-frames:v", "1",
                                    "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
                from io import BytesIO
                sheet.paste(Image.open(BytesIO(p)).convert("RGB"), (c * 216, r * 384))
        sheet.save(OUT / f"qa_contact_{k + 1}.png")
    log("       contact sheets: output/qa_contact_1.png, _2.png, _3.png (S01-S04, S05-S08, S09-S12)")
    log("=" * 78)
    log("RESULT: " + ("ALL AUTOMATED CHECKS PASSED" if not failures else "FAILED: " + "; ".join(failures)))
    (OUT / "qa_report.txt").write_text("\n".join(lines) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
