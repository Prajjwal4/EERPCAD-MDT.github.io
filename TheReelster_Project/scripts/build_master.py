#!/usr/bin/env python3
"""Assemble the delivery files from the 12 scene renders (video timed to the narration; audio is edited by the user).

  output/TheReelster_Master_Captions.mp4         1080x1920, 30 fps, H.264, 1800 frames, captions burned in
  output/TheReelster_Master_NoCaptions.mp4       the same frames without captions
  output/TheReelster_Reference_With_Narration.mp4  captioned master + the ElevenLabs narration (reference only)
  output/voiceover_01.wav                        the narration on the SAME 60 s timeline (drop in at 0:00)
  output/captions.srt                            narration caption cues (global time)
  output/markers.csv                             scene boundaries + every word (seconds and frames) for the editor
Scene files are joined with a stream copy: every scene starts on an IDR frame and shares identical encoding parameters.
"""
import csv
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qa_scene
from config import load_config
from script_plan import Plan

cfg = load_config()
OUT = cfg.path("output")
FPS = cfg.fps


def concat(files, dst: Path):
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{f.resolve()}'\n" for f in files))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(dst)], check=True)
    lst.unlink()


def srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main():
    OUT.mkdir(exist_ok=True)
    scenes = [cfg.path(f"scenes/S{n:02d}.mp4") for n in range(1, 13)]
    cleans = [cfg.path(f"scenes/clean/S{n:02d}_clean.mp4") for n in range(1, 13)]
    for f in scenes + cleans:
        if not f.is_file():
            sys.exit(f"MISSING scene file: {f}")
        if not qa_scene.format_ok(qa_scene.probe(f)):
            sys.exit(f"{f.name} is not 150 frames / 1080x1920 / 30 fps")
    cap, nocap = OUT / "TheReelster_Master_Captions.mp4", OUT / "TheReelster_Master_NoCaptions.mp4"
    concat(scenes, cap)
    concat(cleans, nocap)
    ref = OUT / "TheReelster_Reference_With_Narration.mp4"
    wav = OUT / "voiceover_01.wav"
    shutil.copyfile(cfg.path("assets/audio/voiceover_01.wav"), wav)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(cap), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", "60", "-movflags", "+faststart", str(ref)], check=True)

    plan = Plan()
    # ---- captions.srt: one cue per caption group (<=7 words), global time
    cues = []
    for g in plan._groups:
        cues.append((g["start"] / FPS, g["end"] / FPS, g["text"]))
    cues.sort()
    with open(OUT / "captions.srt", "w") as f:
        for i, (a, b, t) in enumerate(cues, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n")
    # ---- markers.csv: scene boundaries + every word
    with open(OUT / "markers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "scene", "text", "start_s", "end_s", "start_frame", "end_frame"])
        for s in plan.raw_scenes:
            n = s["number"]
            w.writerow(["scene", s["id"], s["title"], (n - 1) * 5, n * 5, (n - 1) * 150, n * 150])
        for ln in plan.timing["lines"]:
            w.writerow(["line", f"S{ln['scene']:02d}", ln["text"], ln["start"], ln["end"], ln["start_frame"], ln["end_frame"]])
            for wd in ln["words"]:
                w.writerow(["word", f"S{ln['scene']:02d}", wd["word"], wd["start"], wd["end"], wd["start_frame"], wd["end_frame"]])
    for p in (cap, nocap, ref):
        pr = qa_scene.probe(p) if p != ref else None
        print(p.name, "->", f"{pr['width']}x{pr['height']} {pr['r_frame_rate']} frames={pr['nb_read_frames']} dur={pr['duration']:.3f}s" if pr else "ok")
    print("wrote", ", ".join(x.name for x in (wav, OUT / 'captions.srt', OUT / 'markers.csv')))


if __name__ == "__main__":
    main()
